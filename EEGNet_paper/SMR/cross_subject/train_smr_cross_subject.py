import argparse, os, csv, random
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.callbacks import ModelCheckpoint

import sys

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
EEGMODELS_DIR = os.path.abspath(os.path.join(THIS_DIR, "..", "..", "arl-eegmodels"))
sys.path.insert(0, EEGMODELS_DIR)

from EEGModels import EEGNet


def set_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def build_eegnet(nb_classes, Chans, Samples, dropout):
    model = EEGNet(
        nb_classes=nb_classes,
        Chans=Chans,
        Samples=Samples,
        dropoutRate=dropout,
        kernLength=32,  
        F1=8,
        D=2,
        F2=16,
        dropoutType="Dropout"
    )
    model.compile(
        optimizer=Adam(),  
        loss=SparseCategoricalCrossentropy(),  
        metrics=["accuracy"]
    )
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_npz", required=True)
    ap.add_argument("--test_npz", required=True)
    ap.add_argument("--outdir", default="results_smr")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--batch_size", type=int, default=16)  
    ap.add_argument("--reps", type=int, default=10)         # 10 reps par sujet test donc 90 folds
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    set_seeds(args.seed)
    rng = np.random.default_rng(args.seed)

    tr = np.load(args.train_npz)
    te = np.load(args.test_npz)

    Xtr = tr["X"][..., np.newaxis].astype(np.float32)
    ytr = tr["y"].astype(np.int64)
    subj_tr = tr["subject"].astype(np.int64)

    Xte = te["X"][..., np.newaxis].astype(np.float32)
    yte = te["y"].astype(np.int64)
    subj_te = te["subject"].astype(np.int64)

    nb_classes = 4
    Chans = Xtr.shape[1]
    Samples = Xtr.shape[2]

    test_subjects = np.unique(subj_te)
    train_subjects = np.unique(subj_tr)

    rows = []
    summary = []

    for test_s in test_subjects:
        idx_test = np.where(subj_te == test_s)[0]
        X_test, y_test = Xte[idx_test], yte[idx_test]

        # 8 autres sujets train pour composer train=5, val=3 
        others = [int(s) for s in train_subjects if int(s) != int(test_s)]
        rep_acc = []

        for rep in range(args.reps):
            # tirage aleatoire 5 train + 3 val
            perm = rng.permutation(others)
            train_subs = perm[:5]
            val_subs = perm[5:8]

            idx_train = np.where(np.isin(subj_tr, train_subs))[0]
            idx_val = np.where(np.isin(subj_tr, val_subs))[0]

            X_train, y_train = Xtr[idx_train], ytr[idx_train]
            X_val, y_val = Xtr[idx_val], ytr[idx_val]

            tf.keras.backend.clear_session()
            model = build_eegnet(nb_classes, Chans, Samples, dropout=0.25)  

            ckpt_path = os.path.join(
                args.outdir, f"cross_testS{int(test_s):02d}_rep{rep:02d}.weights.h5"
            )
            ckpt = ModelCheckpoint(
                ckpt_path,
                monitor="val_loss",
                save_best_only=True,
                save_weights_only=True,
                mode="min",
                verbose=0
            )

            model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=args.epochs,
                batch_size=args.batch_size,
                verbose=0,
                callbacks=[ckpt]
            )

            model.load_weights(ckpt_path)
            _, acc = model.evaluate(X_test, y_test, verbose=0)
            acc = float(acc)

            rep_acc.append(acc)
            rows.append({
                "test_subject": int(test_s),
                "rep": int(rep),
                "acc": acc,
                "ckpt": os.path.basename(ckpt_path),
                "train_subjects": ",".join(map(str, train_subs.tolist())),
                "val_subjects": ",".join(map(str, val_subs.tolist())),
            })

        mean_s = float(np.mean(rep_acc))
        summary.append({"test_subject": int(test_s), "acc_mean": mean_s})
        print(f"Test subject {int(test_s):02d}  mean over {args.reps} reps = {mean_s:.4f}")

    out_csv = os.path.join(args.outdir, "cross_subject_90fold.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["test_subject", "rep", "acc", "ckpt", "train_subjects", "val_subjects"]
        )
        w.writeheader()
        w.writerows(rows)

    out_sum = os.path.join(args.outdir, "cross_subject_summary.csv")
    with open(out_sum, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["test_subject", "acc_mean"])
        w.writeheader()
        w.writerows(summary)

    all_acc = np.array([r["acc"] for r in rows], dtype=float)
    mean_acc = float(all_acc.mean())
    std_acc = float(all_acc.std(ddof=1))
    sem_acc = float(std_acc / np.sqrt(len(all_acc)))

    print("\nCROSS-SUBJECT (90 folds)")
    print("Mean acc:", mean_acc)
    print("Std acc :", std_acc)
    print("SEM     :", sem_acc)
    print("2*SEM   :", 2 * sem_acc)
    print(f"\nSaved: {out_csv}")
    print(f"Saved: {out_sum}")
    print(f"Saved weights in: {args.outdir} (cross_testSXX_repYY.weights.h5)")


if __name__ == "__main__":
    main()
