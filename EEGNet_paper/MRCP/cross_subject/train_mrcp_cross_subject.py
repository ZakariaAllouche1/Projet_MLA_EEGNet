import argparse, os, sys, csv, glob, random
from math import ceil
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import to_categorical

# Utils
def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def ensure_nct(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X)
    if X.ndim == 3:
        return X
    if X.ndim == 4:
        if X.shape[1] == 1:      # (N,1,C,T)
            return X[:, 0, :, :]
        if X.shape[-1] == 1:     # (N,C,T,1)
            return X[:, :, :, 0]
    raise ValueError(f"Unexpected X shape {X.shape}. Expected (N,C,T) or (N,1,C,T) or (N,C,T,1).")

def format_for_model(X_nct: np.ndarray, model_input_shape: tuple) -> np.ndarray:
    ish = tuple(model_input_shape)
    if len(ish) != 4:
        raise ValueError(f"Unsupported model input shape: {ish}")

    # channels_last : (None, C, T, 1)
    if ish[-1] == 1:
        return X_nct[..., np.newaxis].astype(np.float32)

    # channels_first : (None, 1, C, T)
    if ish[1] == 1:
        return X_nct[:, np.newaxis, :, :].astype(np.float32)

    raise ValueError(f"Unsupported model input shape: {ish}")

def auc_binary(y_true_int: np.ndarray, y_proba_pos: np.ndarray) -> float:
    m = tf.keras.metrics.AUC(curve="ROC")
    m.update_state(y_true_int.astype(np.int32), y_proba_pos.astype(np.float32))
    return float(m.result().numpy())

# EEGNet-8,2 (paper config)
def build_eegnet_82(nb_classes: int, Chans: int, Samples: int, dropout: float, kernLength: int):
    from EEGModels import EEGNet
    model = EEGNet(nb_classes = nb_classes, Chans = Chans, Samples = Samples, dropoutRate = dropout, kernLength = kernLength, F1 = 8, D = 2, F2 = 16, dropoutType = "Dropout",)
    model.compile(optimizer = Adam(), loss = CategoricalCrossentropy(), metrics = ["accuracy"],)
    return model

def load_mrcp_subject_npzs(data_dir: str):
    files = sorted(glob.glob(os.path.join(data_dir, "mrcp_P*_eegnet.npz")))
    if not files:
        raise RuntimeError(f"No MRCP subject NPZ found in {data_dir} (expected mrcp_P*_eegnet.npz)")

    X_list, y_list, subj_list, fs_list = [], [], [], []

    for p in files:
        d = np.load(p, allow_pickle=True)
        if "X" not in d.files or "y" not in d.files:
            raise RuntimeError(f"{os.path.basename(p)} missing X/y keys: {d.files}")

        X = ensure_nct(d["X"]).astype(np.float32)
        y = d["y"].astype(np.int64)

        if "subject" in d.files:
            s = int(np.asarray(d["subject"]).ravel()[0])
        else:
            base = os.path.basename(p)
            s = int(base.split("_")[1][1:])

        fs = float(np.asarray(d["fs"]).ravel()[0]) if "fs" in d.files else 128.0

        if set(np.unique(y).tolist()) - {0, 1}:
            raise ValueError(f"{os.path.basename(p)} has non-binary labels: {np.unique(y)}")

        X_list.append(X)
        y_list.append(y)
        subj_list.append(np.full((len(y),), s, dtype = np.int64))
        fs_list.append(fs)

    # check consistent shapes
    C0, T0 = X_list[0].shape[1], X_list[0].shape[2]
    for i, X in enumerate(X_list):
        if X.shape[1] != C0 or X.shape[2] != T0:
            raise ValueError(
                f"Inconsistent X shapes across subjects. "
                f"First is (C,T)=({C0},{T0}); file #{i} has (C,T)=({X.shape[1]},{X.shape[2]})."
            )

    X_all = np.concatenate(X_list, axis = 0)
    y_all = np.concatenate(y_list, axis = 0)
    subj_all = np.concatenate(subj_list, axis = 0)

    fs_med = float(np.median(np.array(fs_list, dtype = float)))
    return X_all, y_all, subj_all, fs_med, files

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required = True, help = "Directory containing mrcp_P*_eegnet.npz")
    ap.add_argument("--eegmodels_dir", required = True, help = "Path to arl-eegmodels (contains EEGModels.py)")
    ap.add_argument("--outdir", default="results_mrcp_cross_EEGNet82")
    ap.add_argument("--seed", type = int, default = 42)
    ap.add_argument("--reps", type = int, default = 30, help = "Paper: 30 folds")
    ap.add_argument("--epochs", type = int, default = 500, help = "Paper: 500 epochs + validation stopping (best val_loss)")
    ap.add_argument("--batch_size", type = int, default = 16)
    ap.add_argument("--verbose", type = int, default = 0)
    ap.add_argument("--cpu_only", type = int, default = 0)
    args = ap.parse_args()

    if args.cpu_only == 1:
        tf.config.set_visible_devices([], "GPU")

    os.makedirs(args.outdir, exist_ok = True)

    set_seeds(args.seed)
    rng = np.random.default_rng(args.seed)

    sys.path.insert(0, os.path.abspath(args.eegmodels_dir)) # import EEGModels.py
    X_nct, y_int, subj, fs, files = load_mrcp_subject_npzs(args.data_dir) # load data
    subjects = np.unique(subj)
    kernLength = int(round(fs / 2.0)) # kernLength = fs/2
    dropout = 0.25 # dropout p = 0.25 for cross-subject
    Chans = int(X_nct.shape[1])
    Samples = int(X_nct.shape[2])
    nb_classes = 2

    print("MRCP CROSS-SUBJECT (paper-style)")
    print(f"Loaded subjects: {len(subjects)} -> {subjects.tolist()}")
    print(f"Trials total: {len(y_int)} | X: {X_nct.shape} | fs~{fs:.1f} Hz | kernLength={kernLength} | dropout={dropout}")
    print("Protocol (paper): each rep picks 4 val subjects + 1 test subject at random; remaining subjects train; repeated 30 times.\n")

    csv_path = os.path.join(args.outdir, "mrcp_cross_subject_folds.csv")
    rows = []
    test_aucs = []

    for rep in range(args.reps):
        # choose 4 validation subjects
        if len(subjects) < 6:
            raise RuntimeError("Need at least 6 subjects for 4 val + 1 test + >=1 train.")
        val_subs = rng.choice(subjects, size=4, replace=False)

        remaining = np.array([s for s in subjects if s not in set(val_subs)], dtype=subjects.dtype)
        test_sub = int(rng.choice(remaining, size = 1, replace = False)[0])
        train_subs = np.array([s for s in remaining if int(s) != test_sub], dtype=subjects.dtype)

        idx_train = np.where(np.isin(subj, train_subs))[0]
        idx_val   = np.where(np.isin(subj, val_subs))[0]
        idx_test  = np.where(subj == test_sub)[0]

        Xtr_nct, ytr = X_nct[idx_train], y_int[idx_train]
        Xva_nct, yva = X_nct[idx_val],   y_int[idx_val]
        Xte_nct, yte = X_nct[idx_test],  y_int[idx_test]

        ytr_oh = to_categorical(ytr, nb_classes).astype(np.float32)
        yva_oh = to_categorical(yva, nb_classes).astype(np.float32)

        tf.keras.backend.clear_session()
        model = build_eegnet_82(nb_classes, Chans, Samples, dropout=dropout, kernLength=kernLength)

        # format inputs for model (channels_first vs channels_last)
        Xtr = format_for_model(Xtr_nct, model.input_shape)
        Xva = format_for_model(Xva_nct, model.input_shape)
        Xte = format_for_model(Xte_nct, model.input_shape)

        # save best weights at lowest val_loss
        w_path = os.path.join(args.outdir, f"mrcp_cross_rep{rep:02d}_testS{test_sub:02d}.weights.h5")
        ckpt = ModelCheckpoint(w_path, monitor = "val_loss", save_best_only = True, save_weights_only = True, mode = "min", verbose = 0)

        model.fit(Xtr, ytr_oh, validation_data = (Xva, yva_oh), epochs = args.epochs, batch_size = args.batch_size, verbose = args.verbose, callbacks = [ckpt])

        model.load_weights(w_path)

        val_proba = model.predict(Xva, verbose = 0)[:, 1]
        test_proba = model.predict(Xte, verbose = 0)[:, 1]
        val_auc = auc_binary(yva, val_proba)
        test_auc = auc_binary(yte, test_proba)
        test_aucs.append(test_auc)

        rows.append({
            "rep": rep,
            "train_subjects": " ".join(map(str, train_subs.tolist())),
            "val_subjects": " ".join(map(str, val_subs.tolist())),
            "test_subject": int(test_sub),
            "n_train": int(len(idx_train)),
            "n_val": int(len(idx_val)),
            "n_test": int(len(idx_test)),
            "val_auc": float(val_auc),
            "test_auc": float(test_auc),
            "weights": os.path.basename(w_path),
        })

        print(f"[MRCP cross] rep {rep+1:02d}/{args.reps} | test=S{test_sub:02d} | val_auc={val_auc:.4f} | test_auc={test_auc:.4f}")

    # write CSV
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    test_aucs = np.array(test_aucs, dtype=float)
    mean_auc = float(test_aucs.mean())
    std_auc = float(test_aucs.std(ddof=1)) if len(test_aucs) > 1 else 0.0
    sem_auc = float(std_auc / np.sqrt(len(test_aucs))) if len(test_aucs) > 1 else 0.0

    print("\nMRCP CROSS-SUBJECT — summary over folds (paper-style)")
    print(f"Mean test AUC = {mean_auc:.6f}")
    print(f"Std  test AUC = {std_auc:.6f}")
    print(f"SEM  test AUC = {sem_auc:.6f}")
    print(f"\nSaved CSV: {csv_path}")
    print(f"Saved weights in: {args.outdir}/")

if __name__ == "__main__":
    main()