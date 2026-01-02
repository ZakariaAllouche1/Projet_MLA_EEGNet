import os
import sys
import argparse
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import roc_auc_score
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS_DIR)

from EEGModels import EEGNet

# Utils
def set_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def class_weight_paper(y):
    y = y.astype(int)
    counts = np.bincount(y)
    maj = counts.max()
    maj_class = counts.argmax()

    cw = {}
    for c, n in enumerate(counts):
        if c == maj_class:
            cw[c] = 1.0
        else:
            cw[c] = float(np.ceil(maj / n))
    return cw


def build_eegnet_ern(chans, samples, dropout):
    model = EEGNet(nb_classes = 2, Chans = chans, Samples = samples, dropoutRate = dropout, kernLength = 64, F1 = 8, D = 2, F2 = 16, dropoutType = "Dropout")
    model.compile(optimizer = Adam(), loss = CategoricalCrossentropy(), metrics=["accuracy"])
    return model

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_npz", required = True)
    ap.add_argument("--out", default = "results_ern_within")
    ap.add_argument("--seed", type = int, default = 42)
    ap.add_argument("--epochs", type = int, default = 500)
    ap.add_argument("--batch_size", type = int, default = 16)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok = True)
    set_seeds(args.seed)

    d = np.load(args.train_npz, allow_pickle = True)

    X = d["X"].astype(np.float32) # (N, 56, 160)
    y = d["y"].astype(np.int64) # (N,)
    subject = d["subject"].astype(np.int64)
    session = d["session"].astype(np.int64)
    id_feedback = d["id_feedback"]

    # EEGNet expects (N, Chans, Samples, 1)
    X = X[..., np.newaxis]

    chans = X.shape[1]
    samples = X.shape[2]

    y_oh = tf.keras.utils.to_categorical(y, 2).astype(np.float32)

    subjects = np.unique(subject)
    rows = []

    for s in subjects:
        idx_s = np.where(subject == s)[0]

        # blockwise 4-fold CV
        order = np.lexsort((id_feedback[idx_s], session[idx_s]))
        idx_s = idx_s[order]

        N = len(idx_s)
        cuts = np.linspace(0, N, 5).astype(int)
        block4 = np.empty(N, dtype=int)
        for b in range(4):
            block4[cuts[b]:cuts[b + 1]] = b

        fold_aucs = []

        for test_b in range(4):
            val_b = (test_b + 1) % 4
            train_bs = [b for b in range(4) if b not in (test_b, val_b)]

            idx_train = idx_s[np.isin(block4, train_bs)]
            idx_val   = idx_s[block4 == val_b]
            idx_test  = idx_s[block4 == test_b]

            X_train, y_train = X[idx_train], y_oh[idx_train]
            X_val, y_val     = X[idx_val],   y_oh[idx_val]
            X_test           = X[idx_test]
            y_test           = y[idx_test]

            tf.keras.backend.clear_session()

            model = build_eegnet_ern(chans, samples, dropout=0.5)

            cw = class_weight_paper(y[idx_train])

            ckpt_path = os.path.join(args.out, f"ern_within_s{s:02d}_fold{test_b}.weights.h5")
            ckpt = ModelCheckpoint(ckpt_path, monitor = "val_loss", save_best_only = True, save_weights_only = True, mode = "min", verbose = 0)
            model.fit(X_train, y_train, validation_data = (X_val, y_val), epochs = args.epochs, batch_size = args.batch_size, verbose = 0, callbacks = [ckpt], class_weight = cw)
            model.load_weights(ckpt_path)
            proba = model.predict(X_test, verbose=0)[:, 1]
            auc = float(roc_auc_score(y_test, proba))
            fold_aucs.append(auc)
            rows.append({"subject": int(s), "fold": int(test_b + 1), "auc": auc})

        print(
            f"Subject {int(s):02d} | mean_AUC={np.mean(fold_aucs):.4f} "
            f"| folds={np.round(fold_aucs,4)}")

    df = pd.DataFrame(rows)
    out_csv = os.path.join(args.out, "ern_within_subject_4fold_auc.csv")
    df.to_csv(out_csv, index=False)

    print("\\nWITHIN-SUBJECT ERN (AUC)")
    print("Mean AUC over subjects:", df.groupby("subject")["auc"].mean().mean())
    print("Std  AUC over subjects:", df.groupby("subject")["auc"].mean().std())
    print("Saved:", out_csv)


if __name__ == "__main__":
    main()