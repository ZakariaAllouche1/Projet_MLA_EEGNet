#!/usr/bin/env python3
"""
EEGNet (Lawhern et al., 2018) — P300 WITHIN-SUBJECT

Protocol EXACTLY as described in the paper:
- 4-fold BLOCKWISE cross-validation (4 runs = 4 blocks)
- Per fold: 2 blocks TRAIN / 1 block VAL / 1 block TEST
- EEGNet-8,2
- Dropout = 0.5 (within-subject)
- Adam optimizer (default parameters)
- Categorical cross-entropy
- 500 epochs
- Validation stopping: keep weights with minimum val_loss
- Class-weight for imbalance: majority=1, minority=ceil(maj/min)
- Metric: AUC ROC (computed after training)
"""

import argparse, os, sys, csv
from math import ceil
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import to_categorical


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

def ensure_nct(X):
    """Ensure shape (N, C, T)."""
    X = np.asarray(X)
    if X.ndim == 3:
        return X
    if X.ndim == 4:
        if X.shape[1] == 1:
            return X[:, 0, :, :]
        if X.shape[-1] == 1:
            return X[:, :, :, 0]
    raise ValueError(f"Unexpected X shape {X.shape}")

def format_for_model(X_nct, model_input_shape):
    """Match EEGModels.py layout."""
    ish = tuple(model_input_shape)
    if ish[-1] == 1:
        return X_nct[..., np.newaxis].astype(np.float32)
    if ish[1] == 1:
        return X_nct[:, np.newaxis, :, :].astype(np.float32)
    raise ValueError(f"Unsupported model input shape {ish}")

def class_weight_paper(y):
    """Majority=1, minority=ceil(n_major/n_minor)."""
    y = np.asarray(y).astype(int)
    classes, counts = np.unique(y, return_counts=True)
    if len(classes) <= 1:
        return {int(classes[0]): 1.0}
    n_major = counts.max()
    major_class = classes[counts.argmax()]
    cw = {}
    for c, n in zip(classes, counts):
        if c == major_class:
            cw[int(c)] = 1.0
        else:
            cw[int(c)] = float(ceil(n_major / max(1, n)))
    return cw

def auc_roc(y_true, y_score):
    m = tf.keras.metrics.AUC(curve="ROC")
    m.update_state(y_true.astype(np.int32), y_score.astype(np.float32))
    return float(m.result().numpy())


# ---------------------------------------------------------------------
# EEGNet-8,2 (paper configuration)
# ---------------------------------------------------------------------

def build_eegnet(nb_classes, Chans, Samples, dropout, kernLength):
    from EEGModels import EEGNet
    model = EEGNet(
        nb_classes=nb_classes,
        Chans=Chans,
        Samples=Samples,
        dropoutRate=dropout,
        kernLength=kernLength,
        F1=8,
        D=2,
        F2=16,
        dropoutType="Dropout",
    )
    model.compile(
        optimizer=Adam(),
        loss=CategoricalCrossentropy(),
        metrics=["accuracy"],
    )
    return model


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_npz", required=True,
                    help="P300 NPZ with keys X,y,subject,run")
    ap.add_argument("--eegmodels_dir", required=True,
                    help="Path to arl-eegmodels (EEGModels.py)")
    ap.add_argument("--outdir", default="results_p300_within_EEGNet82")
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--verbose", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    sys.path.insert(0, args.eegmodels_dir)

    d = np.load(args.data_npz, allow_pickle=True)

    X = ensure_nct(d["X"]).astype(np.float32)
    y = d["y"].astype(np.int64)
    subject = d["subject"].astype(np.int64)
    run = d["run"].astype(np.int64)

    nb_classes = 2
    Chans, Samples = X.shape[1], X.shape[2]

    fs = float(d["fs"]) if "fs" in d.files else 128.0
    kernLength = int(fs / 2)

    subjects = np.unique(subject)

    csv_path = os.path.join(args.outdir, "p300_within_subject_4fold_auc.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow([
            "subject", "fold",
            "train_runs", "val_run", "test_run",
            "val_auc", "test_auc"
        ])

        for s in subjects:
            idx_s = np.where(subject == s)[0]
            Xs, ys, rs = X[idx_s], y[idx_s], run[idx_s]

            runs = np.sort(np.unique(rs))
            if len(runs) != 4:
                raise ValueError(f"Subject {s}: expected exactly 4 runs")

            for fold in range(4):
                test_run = runs[fold]
                val_run = runs[(fold + 1) % 4]
                train_runs = [r for r in runs if r not in (test_run, val_run)]

                idx_tr = np.where(np.isin(rs, train_runs))[0]
                idx_va = np.where(rs == val_run)[0]
                idx_te = np.where(rs == test_run)[0]

                Xtr, ytr = Xs[idx_tr], ys[idx_tr]
                Xva, yva = Xs[idx_va], ys[idx_va]
                Xte, yte = Xs[idx_te], ys[idx_te]

                cw = class_weight_paper(ytr)

                ytr_oh = to_categorical(ytr, nb_classes)
                yva_oh = to_categorical(yva, nb_classes)
                yte_oh = to_categorical(yte, nb_classes)

                tf.keras.backend.clear_session()
                model = build_eegnet(
                    nb_classes, Chans, Samples,
                    dropout=0.5,
                    kernLength=kernLength
                )

                Xtr_f = format_for_model(Xtr, model.input_shape)
                Xva_f = format_for_model(Xva, model.input_shape)
                Xte_f = format_for_model(Xte, model.input_shape)

                wpath = os.path.join(
                    args.outdir, f"p300_within_s{s:02d}_fold{fold}.weights.h5"
                )

                ckpt = ModelCheckpoint(
                    wpath,
                    monitor="val_loss",
                    save_best_only=True,
                    save_weights_only=True,
                    mode="min",
                    verbose=0
                )

                model.fit(
                    Xtr_f, ytr_oh,
                    validation_data=(Xva_f, yva_oh),
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    class_weight=cw,
                    callbacks=[ckpt],
                    verbose=args.verbose
                )

                model.load_weights(wpath)

                val_auc = auc_roc(yva, model.predict(Xva_f, verbose=0)[:, 1])
                test_auc = auc_roc(yte, model.predict(Xte_f, verbose=0)[:, 1])

                writer.writerow([
                    int(s), int(fold),
                    " ".join(map(str, train_runs)),
                    int(val_run), int(test_run),
                    f"{val_auc:.6f}", f"{test_auc:.6f}"
                ])
                fcsv.flush()


if __name__ == "__main__":
    main()
