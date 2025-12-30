import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
    balanced_accuracy_score,
    cohen_kappa_score,
)

from models_eegnet import EEGNetMSSE


# --------------------------
# Utils
# --------------------------

def set_all_seeds(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Tee:
    def __init__(self, file_path: Path):
        self.file = open(file_path, "w", encoding="utf-8")
        self._stdout = None

    def __enter__(self):
        import sys
        self._stdout = sys.stdout
        sys.stdout = self
        return self

    def __exit__(self, exc_type, exc, tb):
        import sys
        sys.stdout = self._stdout
        self.file.close()

    def write(self, data):
        self._stdout.write(data)
        self.file.write(data)

    def flush(self):
        self._stdout.flush()
        self.file.flush()


# --------------------------
# Dataset
# --------------------------

class EEGDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.int64))

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


@dataclass
class Split:
    idx_train: np.ndarray
    idx_val: np.ndarray
    idx_test: np.ndarray
    train_subjects: List[int]
    val_subjects: List[int]
    test_subjects: List[int]
    fold_id: int
    seed: int
    protocol_tag: str


# --------------------------
# Splits
# --------------------------

def _contiguous_blocks(indices: np.ndarray, n_blocks: int = 4) -> List[np.ndarray]:
    blocks = np.array_split(indices, n_blocks)
    return [b.astype(np.int64) for b in blocks if len(b) > 0]


def make_within_subject_blockwise_splits(subject_arr: np.ndarray,
                                        subject_id: int,
                                        y: np.ndarray,
                                        seed: int = 0) -> List[Split]:
    idx_sub = np.where(subject_arr == subject_id)[0].astype(np.int64)
    if len(idx_sub) < 8:
        raise ValueError(f"Sujet {subject_id}: trop peu d'essais ({len(idx_sub)}) pour 4 blocs.")

    idx_sub = np.sort(idx_sub)

    def build_blocks(order_idx: np.ndarray) -> List[np.ndarray]:
        blocks = _contiguous_blocks(order_idx, n_blocks=4)
        if len(blocks) != 4:
            raise ValueError(f"Sujet {subject_id}: impossible de construire 4 blocs (n={len(order_idx)}).")
        return blocks

    blocks = build_blocks(idx_sub)

    mono = []
    for bi, b in enumerate(blocks):
        cls = np.unique(y[b])
        mono.append(len(cls) == 1)

    if any(mono):
        counts = dict(zip(*np.unique(y[idx_sub], return_counts=True)))
        print(f"[WARN] WITHIN blockwise: sujet {subject_id} bloc(s) mono-classe. "
              f"Distrib labels sujet={counts} → shuffle déterministe pour construire les blocs.")
        rng = np.random.RandomState(seed + int(subject_id))
        shuffled = rng.permutation(idx_sub)
        blocks = build_blocks(shuffled)

    splits: List[Split] = []
    for fold_id in range(4):
        test_b = blocks[fold_id]
        val_b = blocks[(fold_id + 1) % 4]
        train_b = np.concatenate([blocks[(fold_id + 2) % 4], blocks[(fold_id + 3) % 4]], axis=0)

        splits.append(Split(
            idx_train=train_b, idx_val=val_b, idx_test=test_b,
            train_subjects=[int(subject_id)], val_subjects=[int(subject_id)], test_subjects=[int(subject_id)],
            fold_id=fold_id, seed=seed,
            protocol_tag=f"within_sub{subject_id}_fold{fold_id}"
        ))
    return splits


def make_cross_subject_mrcp_folds(subject_arr: np.ndarray, n_folds: int, seed: int) -> List[Split]:
    rng = np.random.RandomState(seed)
    subjects = np.unique(subject_arr).astype(int).tolist()
    if len(subjects) < 6:
        raise ValueError("Cross-subject MRCP/P300 nécessite >= 6 sujets (1 test, 4 val, >=1 train).")

    splits: List[Split] = []
    for fold_id in range(n_folds):
        perm = rng.permutation(subjects)
        test_sub = int(perm[0])
        val_subs = [int(s) for s in perm[1:5]]
        train_subs = [int(s) for s in perm[5:]]

        idx_test = np.where(subject_arr == test_sub)[0]
        idx_val = np.where(np.isin(subject_arr, val_subs))[0]
        idx_train = np.where(np.isin(subject_arr, train_subs))[0]

        splits.append(Split(
            idx_train=idx_train.astype(np.int64),
            idx_val=idx_val.astype(np.int64),
            idx_test=idx_test.astype(np.int64),
            train_subjects=train_subs, val_subjects=val_subs, test_subjects=[test_sub],
            fold_id=fold_id, seed=seed,
            protocol_tag=f"cross_fold{fold_id}_test{test_sub}"
        ))
    return splits


def make_cross_subject_smr_folds(subject_train: np.ndarray,
                                subject_test: np.ndarray,
                                n_repeats_per_subject: int,
                                seed: int) -> List[Split]:
    rng = np.random.RandomState(seed)
    subjects = np.unique(subject_train).astype(int).tolist()
    if len(subjects) != 9:
        print(f"[WARN] SMR: nombre de sujets train détecté = {len(subjects)} (attendu 9).")

    splits: List[Split] = []
    fold_id = 0
    for test_sub in subjects:
        others = [s for s in subjects if s != test_sub]
        if len(others) < 8:
            raise ValueError("SMR cross-subject nécessite 9 sujets (8 autres pour train/val).")
        for r in range(n_repeats_per_subject):
            perm = rng.permutation(others)
            train_subs = [int(s) for s in perm[:5]]
            val_subs = [int(s) for s in perm[5:8]]

            idx_train = np.where(np.isin(subject_train, train_subs))[0]
            idx_val = np.where(np.isin(subject_train, val_subs))[0]
            idx_test = np.where(subject_test == test_sub)[0]

            splits.append(Split(
                idx_train=idx_train.astype(np.int64),
                idx_val=idx_val.astype(np.int64),
                idx_test=idx_test.astype(np.int64),
                train_subjects=train_subs, val_subjects=val_subs, test_subjects=[int(test_sub)],
                fold_id=fold_id, seed=seed,
                protocol_tag=f"smr_cross_test{test_sub}_rep{r}"
            ))
            fold_id += 1
    return splits


# --------------------------
# Modèle
# --------------------------

def build_model(dataset: str, protocol: str,
                n_channels: int, n_times: int, n_classes: int) -> nn.Module:
    pdrop = 0.5 if protocol == "within" else 0.25

    kernel_len = 32 if dataset == "smr" else 64

    return EEGNetMSSE(
        n_channels=n_channels,
        n_times=n_times,
        n_classes=n_classes,
        F1_branch=4,
        D=2,
        kernel_len_short=32,
        kernel_len_long=kernel_len,
        dropout_p=pdrop,
        se_reduction=4
    )



# --------------------------
# Train / Eval
# --------------------------

def compute_class_weights(y_train: np.ndarray) -> Optional[torch.Tensor]:
    classes, counts = np.unique(y_train, return_counts=True)
    if len(classes) <= 1:
        return None
    max_count = counts.max()
    weights = []
    for c, cnt in zip(classes, counts):
        if cnt == max_count:
            weights.append(1.0)
        else:
            ratio = max_count / cnt
            weights.append(float(np.ceil(ratio)))
    w = np.zeros(int(classes.max()) + 1, dtype=np.float32)
    for c, wt in zip(classes, weights):
        w[int(c)] = wt
    return torch.tensor(w, dtype=torch.float32)


def train_one_fold(model: nn.Module,
                   train_loader: DataLoader,
                   val_loader: DataLoader,
                   device: torch.device,
                   lr: float,
                   weight_decay: float,
                   max_epochs: int,
                   class_weights: Optional[torch.Tensor],
                   log_every: int = 0,
                   log_best_only: bool = False,
                   log_first_last: bool = False) -> Dict:
    if class_weights is not None:
        class_weights = class_weights.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_loss = float("inf")
    best_state = None
    best_epoch = -1

    for epoch in range(1, max_epochs + 1):
        model.train()
        tr_losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            tr_losses.append(loss.item())

        model.eval()
        va_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                loss = criterion(logits, yb)
                va_losses.append(loss.item())

        mean_tr = float(np.mean(tr_losses)) if tr_losses else float("nan")
        mean_va = float(np.mean(va_losses)) if va_losses else float("nan")

        is_best = mean_va < best_val_loss - 1e-6

        should_print = False
        if log_first_last and (epoch == 1 or epoch == max_epochs):
            should_print = True
        if log_every and (epoch % log_every == 0):
            should_print = True
        if log_best_only and is_best:
            should_print = True

        if should_print:
            tag = " *best*" if is_best else ""
            print(f"Epoch {epoch:03d} | tr_loss={mean_tr:.4f} va_loss={mean_va:.4f}{tag}")

        if is_best:
            best_val_loss = mean_va
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state, strict=True)

    return {
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch
    }


@torch.no_grad()
def predict_pack(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, np.ndarray]:
    model.eval()

    y_true_list = []
    y_pred_list = []
    prob1_list = []

    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        logits = model(xb)
        probs = torch.softmax(logits, dim=1)

        y_true_list.append(yb.numpy())
        y_pred_list.append(probs.argmax(dim=1).cpu().numpy())

        if probs.shape[1] == 2:
            prob1_list.append(probs[:, 1].detach().cpu().numpy().astype(np.float16))
        else:
            prob1_list.append(probs.detach().cpu().numpy().astype(np.float16))

    y_true = np.concatenate(y_true_list, axis=0)
    y_pred = np.concatenate(y_pred_list, axis=0)

    if len(prob1_list) > 0:
        probs_out = np.concatenate(prob1_list, axis=0)
    else:
        probs_out = None

    return {"y_true": y_true, "y_pred": y_pred, "probs": probs_out}


def classification_metrics(y_true: np.ndarray,
                           y_pred: np.ndarray,
                           probs: Optional[np.ndarray],
                           n_classes: int) -> Dict:
    out: Dict = {}

    out["accuracy"] = float(accuracy_score(y_true, y_pred))
    out["balanced_accuracy"] = float(balanced_accuracy_score(y_true, y_pred))
    out["kappa"] = float(cohen_kappa_score(y_true, y_pred))

    avg = "macro" if n_classes > 2 else "binary"
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=avg, zero_division=0)
    out["precision"] = float(prec)
    out["recall"] = float(rec)
    out["f1"] = float(f1)

    prec_c, rec_c, f1_c, sup_c = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)
    out["per_class"] = {
        str(i): {
            "precision": float(prec_c[i]),
            "recall": float(rec_c[i]),
            "f1": float(f1_c[i]),
            "support": int(sup_c[i])
        }
        for i in range(len(prec_c))
    }

    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))
    out["confusion_matrix"] = cm.tolist()

    from sklearn.metrics import roc_auc_score

    out["auc"] = None
    out["auc_ovr_macro"] = None

    if probs is not None:
        probs32 = probs.astype(np.float32)

        try:
            if n_classes == 2:
                if len(np.unique(y_true)) == 2:
                    if probs32.ndim == 1:
                        out["auc"] = float(roc_auc_score(y_true, probs32))
                    else:
                        out["auc"] = float(roc_auc_score(y_true, probs32[:, 1]))
            else:
                aucs = []
                for k in range(n_classes):
                    yk = (y_true == k).astype(int)
                    if yk.min() == 0 and yk.max() == 1:
                        aucs.append(float(roc_auc_score(yk, probs32[:, k])))
                if len(aucs) > 0:
                    out["auc_ovr_macro"] = float(np.mean(aucs))
        except Exception:
            pass

    out["sklearn_report"] = classification_report(
        y_true, y_pred, labels=list(range(n_classes)), digits=4, zero_division=0
    )

    return out


def save_row(csv_path: Path, row: Dict):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


# --------------------------
# Main runner
# --------------------------

def load_npz(path: Path) -> Dict[str, np.ndarray]:
    npz = np.load(path, allow_pickle=True)
    return {k: npz[k] for k in npz.files}


def main():
    p = argparse.ArgumentParser()

    p.add_argument("--dataset", choices=["ern", "smr", "mrcp", "p300"], required=True)
    p.add_argument("--protocol", choices=["within", "cross"], required=True)

    p.add_argument("--npz-train", type=str, required=True,
                   help="NPZ principal (train). Pour SMR, c'est la session T.")
    p.add_argument("--npz-test", type=str, default=None,
                   help="NPZ test séparé (ex: SMR session E, ERN Kaggle test).")

    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=2)

    p.add_argument("--n-folds", type=int, default=30, help="MRCP/ERN cross: 30")
    p.add_argument("--smr-repeats", type=int, default=10, help="SMR cross: 10 répétitions par sujet")
    p.add_argument("--seed", type=int, default=2025)

    p.add_argument("--out-dir", type=str, default="runs_paper")
    p.add_argument("--log-dir", type=str, default=None, help="Si défini : exporte les logs console en .txt")
    p.add_argument("--save-best", action="store_true",
                   help="Sauvegarde le meilleur modèle (min val_loss) par fold en .pt dans out-dir/checkpoints/.")

    p.add_argument("--log-every", type=int, default=0,
                   help="Si >0 : print un epoch sur N. Si 0 : pas de print périodique.")
    p.add_argument("--log-best-only", action="store_true",
                   help="Ne print que quand best_val_loss s'améliore (recommandé).")
    p.add_argument("--log-first-last", action="store_true",
                   help="Print aussi epoch 1 et epoch final, même si log-every=0.")

    p.add_argument("--use-class-weights", action="store_true",
                   help="Applique class-weights (papier: P300+ERN). Recommandé pour ERN/P300.")

    args = p.parse_args()
    set_all_seeds(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "results_folds.csv"

    report_dir = out_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)


    log_ctx = None
    if args.log_dir is not None:
        log_dir = Path(args.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"{args.dataset}_{args.protocol}.txt"
        log_ctx = Tee(log_file)

    def _run():
        train_npz = load_npz(Path(args.npz_train))
        Xtr = train_npz["X"]
        ytr = train_npz.get("y", None)
        subj_tr = train_npz["subject"].astype(int)

        if ytr is None:
            raise ValueError("npz-train doit contenir y (labels).")

        n_channels = Xtr.shape[1]
        n_times = Xtr.shape[2]
        n_classes = int(len(np.unique(ytr)))

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Device:", device)
        print(f"Train NPZ: {args.npz_train} | X={Xtr.shape} | classes={np.unique(ytr)}")

        Xte = yte = subj_te = None
        if args.npz_test is not None:
            test_npz = load_npz(Path(args.npz_test))
            Xte = test_npz["X"]
            subj_te = test_npz["subject"].astype(int)
            yte = test_npz.get("y", None)
            print(f"Test NPZ:  {args.npz_test} | X={Xte.shape} | y={'OK' if yte is not None else 'MISSING'}")

        splits: List[Split] = []
        if args.protocol == "within":
            for s in np.unique(subj_tr).astype(int):
                splits.extend(make_within_subject_blockwise_splits(subj_tr, int(s), ytr, seed=args.seed))
        else:
            if args.dataset == "smr":
                if Xte is None or yte is None:
                    raise ValueError("SMR cross nécessite --npz-test avec labels (session E).")
                splits = make_cross_subject_smr_folds(subj_tr, subj_te, args.smr_repeats, seed=args.seed)

            elif args.dataset == "mrcp":
                splits = make_cross_subject_mrcp_folds(subj_tr, n_folds=args.n_folds, seed=args.seed)

            elif args.dataset == "p300":
                if Xte is not None and yte is not None:
                    rng = np.random.RandomState(args.seed)
                    train_subjects = np.unique(subj_tr).astype(int).tolist()
                    test_subjects = np.unique(subj_te).astype(int).tolist()

                    for fold_id in range(args.n_folds):
                        perm = rng.permutation(train_subjects)
                        val_subs = [int(x) for x in perm[:4]]
                        train_subs = [int(x) for x in perm[4:]]

                        idx_train = np.where(np.isin(subj_tr, train_subs))[0]
                        idx_val = np.where(np.isin(subj_tr, val_subs))[0]
                        idx_test = np.arange(len(subj_te))

                        splits.append(Split(
                            idx_train=idx_train.astype(np.int64),
                            idx_val=idx_val.astype(np.int64),
                            idx_test=idx_test.astype(np.int64),
                            train_subjects=train_subs,
                            val_subjects=val_subs,
                            test_subjects=test_subjects,
                            fold_id=fold_id,
                            seed=args.seed,
                            protocol_tag=f"p300_cross_fold{fold_id}_test"
                        ))
                else:
                    splits = make_cross_subject_mrcp_folds(subj_tr, n_folds=args.n_folds, seed=args.seed)

            elif args.dataset == "ern":
                if Xte is not None and yte is not None:
                    test_subjects = np.unique(subj_te).astype(int).tolist()
                    rng = np.random.RandomState(args.seed)
                    train_subjects = np.unique(subj_tr).astype(int).tolist()
                    for fold_id in range(args.n_folds):
                        perm = rng.permutation(train_subjects)
                        val_subs = [int(x) for x in perm[:4]]
                        train_subs = [int(x) for x in perm[4:]]
                        idx_train = np.where(np.isin(subj_tr, train_subs))[0]
                        idx_val = np.where(np.isin(subj_tr, val_subs))[0]
                        idx_test = np.where(np.isin(subj_te, test_subjects))[0]
                        splits.append(Split(
                            idx_train=idx_train.astype(np.int64),
                            idx_val=idx_val.astype(np.int64),
                            idx_test=idx_test.astype(np.int64),
                            train_subjects=train_subs, val_subjects=val_subs, test_subjects=test_subjects,
                            fold_id=fold_id, seed=args.seed,
                            protocol_tag=f"ern_cross_fold{fold_id}"
                        ))
                else:
                    print("[WARN] ERN cross sans labels test. Utilisation d'un split cross-sujet 30 folds (1 test,4 val) sur le NPZ train uniquement.")
                    splits = make_cross_subject_mrcp_folds(subj_tr, n_folds=args.n_folds, seed=args.seed)
            else:
                raise ValueError("Dataset inconnu")

        metric_main = "accuracy" if args.dataset == "smr" else "auc"

        for sp in splits:
            print("\n" + "=" * 72)
            print(f"Fold {sp.fold_id} | {sp.protocol_tag}")
            print(f"Train subjects: {sp.train_subjects}")
            print(f"Val subjects:   {sp.val_subjects}")
            print(f"Test subjects:  {sp.test_subjects}")

            if args.dataset == "smr" and args.protocol == "cross":
                X_train, y_train = Xtr[sp.idx_train], ytr[sp.idx_train]
                X_val, y_val = Xtr[sp.idx_val], ytr[sp.idx_val]
                X_test, y_test = Xte[sp.idx_test], yte[sp.idx_test]
            elif args.dataset == "ern" and args.protocol == "cross" and (Xte is not None and yte is not None):
                X_train, y_train = Xtr[sp.idx_train], ytr[sp.idx_train]
                X_val, y_val = Xtr[sp.idx_val], ytr[sp.idx_val]
                X_test, y_test = Xte[sp.idx_test], yte[sp.idx_test]
            elif args.dataset == "p300" and args.protocol == "cross" and (Xte is not None and yte is not None):
                X_train, y_train = Xtr[sp.idx_train], ytr[sp.idx_train]
                X_val, y_val = Xtr[sp.idx_val], ytr[sp.idx_val]
                X_test, y_test = Xte[sp.idx_test], yte[sp.idx_test]

            else:
                X_train, y_train = Xtr[sp.idx_train], ytr[sp.idx_train]
                X_val, y_val = Xtr[sp.idx_val], ytr[sp.idx_val]
                X_test, y_test = Xtr[sp.idx_test], ytr[sp.idx_test]

            train_loader = DataLoader(
                EEGDataset(X_train, y_train),
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.num_workers,
                pin_memory=True
            )
            val_loader = DataLoader(
                EEGDataset(X_val, y_val),
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=True
            )
            test_loader = DataLoader(
                EEGDataset(X_test, y_test),
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=True
            )

            cw = None
            if args.use_class_weights and args.dataset in ("ern", "p300"):
                cw = compute_class_weights(y_train)

            model = build_model(args.dataset, args.protocol,
                n_channels=n_channels, n_times=n_times, n_classes=n_classes
            ).to(device)

            train_info = train_one_fold(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                device=device,
                lr=args.lr,
                weight_decay=args.weight_decay,
                max_epochs=args.epochs,
                class_weights=cw,
                log_every=args.log_every,
                log_best_only=args.log_best_only,
                log_first_last=args.log_first_last
            )

            if args.save_best:
                ckpt_dir = out_dir / "checkpoints" / args.dataset / args.protocol
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                ckpt_path = ckpt_dir / f"{args.dataset}_{args.protocol}_{sp.protocol_tag}.pt"
                torch.save(
                    {
                        "state_dict": model.state_dict(),
                        "dataset": args.dataset,
                        "protocol": args.protocol,
                        "protocol_tag": sp.protocol_tag,
                        "fold_id": sp.fold_id,
                        "seed": sp.seed,
                        "n_channels": int(n_channels),
                        "n_times": int(n_times),
                        "n_classes": int(n_classes),
                        "best_epoch": int(train_info["best_epoch"]),
                        "best_val_loss": float(train_info["best_val_loss"]),
                    },
                    ckpt_path,
                )

            pred = predict_pack(model, test_loader, device)
            metrics_full = classification_metrics(
                y_true=pred["y_true"],
                y_pred=pred["y_pred"],
                probs=pred["probs"],
                n_classes=n_classes
            )

            main_value = metrics_full["accuracy"] if metric_main == "accuracy" else (
                metrics_full["auc"] if metrics_full.get("auc") is not None else float("nan")
            )

            auc_str = "NA"
            if n_classes == 2:
                if metrics_full.get("auc") is not None:
                    auc_str = f"{metrics_full['auc']:.6f}"
            else:
                if metrics_full.get("auc_ovr_macro") is not None:
                    auc_str = f"{metrics_full['auc_ovr_macro']:.6f}"

            print(
                f"Test | acc={metrics_full['accuracy']:.4f} | bal_acc={metrics_full['balanced_accuracy']:.4f} "
                f"| f1={metrics_full['f1']:.4f} | prec={metrics_full['precision']:.4f} | rec={metrics_full['recall']:.4f} "
                f"| auc={auc_str}"
            )
            print(f"Best val_loss={train_info['best_val_loss']:.6f} @ epoch {train_info['best_epoch']}")

            rep_json = {
                "dataset": args.dataset,
                "protocol": args.protocol,
                "fold_id": sp.fold_id,
                "protocol_tag": sp.protocol_tag,
                "seed": sp.seed,
                "train_subjects": sp.train_subjects,
                "val_subjects": sp.val_subjects,
                "test_subjects": sp.test_subjects,
                "n_train": int(len(X_train)),
                "n_val": int(len(X_val)),
                "n_test": int(len(X_test)),
                "train_info": train_info,
                "metrics": {k: v for k, v in metrics_full.items() if k != "sklearn_report"},
            }

            (report_dir / f"{args.dataset}_{args.protocol}_{sp.protocol_tag}.json").write_text(
                json.dumps(rep_json, indent=2), encoding="utf-8"
            )
            (report_dir / f"{args.dataset}_{args.protocol}_{sp.protocol_tag}_report.txt").write_text(
                metrics_full["sklearn_report"], encoding="utf-8"
            )

            row = {
                "dataset": args.dataset,
                "protocol": args.protocol,
                "fold_id": sp.fold_id,
                "protocol_tag": sp.protocol_tag,
                "seed": sp.seed,
                "train_subjects": json.dumps(sp.train_subjects),
                "val_subjects": json.dumps(sp.val_subjects),
                "test_subjects": json.dumps(sp.test_subjects),
                "n_train": int(len(X_train)),
                "n_val": int(len(X_val)),
                "n_test": int(len(X_test)),
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "batch_size": args.batch_size,
                "epochs": args.epochs,
                "best_epoch": train_info["best_epoch"],
                "best_val_loss": train_info["best_val_loss"],
                "main_metric": metric_main,
                "main_value": main_value,
                "accuracy": metrics_full["accuracy"],
                "balanced_accuracy": metrics_full["balanced_accuracy"],
                "precision": metrics_full["precision"],
                "recall": metrics_full["recall"],
                "f1": metrics_full["f1"],
                "kappa": metrics_full["kappa"],
                "auc": metrics_full["auc"] if metrics_full.get("auc") is not None else "",
                "auc_ovr_macro": metrics_full.get("auc_ovr_macro", ""),
            }
            save_row(csv_path, row)

        print(f"\n[OK] CSV folds: {csv_path}")
        print(f"[OK] Reports dir: {report_dir}")

    if log_ctx is not None:
        with log_ctx:
            _run()
            print("\n[OK] Logs exportés.")
    else:
        _run()


if __name__ == "__main__":
    main()