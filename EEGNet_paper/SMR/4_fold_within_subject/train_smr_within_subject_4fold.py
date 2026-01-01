import argparse, os, csv, random
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
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
        loss=CategoricalCrossentropy(),        
        metrics=["accuracy"]                
    )
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_npz", required=True)
    ap.add_argument("--outdir", default="results_smr")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--batch_size", type=int, default=16)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    set_seeds(args.seed)

    d = np.load(args.train_npz)

    X = d["X"].astype(np.float32)         
    y = d["y"].astype(np.int64)           
    subject = d["subject"].astype(np.int64)
    trial = d["trial"].astype(np.int64)    

    X = X[..., np.newaxis]                 

    nb_classes = 4
    Chans = X.shape[1]
    Samples = X.shape[2]

    y_oh = tf.keras.utils.to_categorical(y, nb_classes).astype(np.float32)

    subjects = np.unique(subject)
    rows = []
    subject_means = []

    for s in subjects:
        idx_s = np.where(subject == s)[0]

        block4 = (trial[idx_s] // 72).astype(np.int64) 

        fold_acc = []
        for test_b in range(4):
            val_b = (test_b + 1) % 4
            train_bs = [b for b in range(4) if b not in (test_b, val_b)]

            idx_train = idx_s[np.isin(block4, train_bs)]
            idx_val = idx_s[block4 == val_b]
            idx_test = idx_s[block4 == test_b]

            X_train, y_train = X[idx_train], y_oh[idx_train]
            X_val, y_val = X[idx_val], y_oh[idx_val]
            X_test, y_test = X[idx_test], y_oh[idx_test]

            tf.keras.backend.clear_session()

            # dropout = 0.5
            model = build_eegnet(nb_classes, Chans, Samples, dropout=0.5)

            ckpt_path = os.path.join(
                args.outdir, f"within_s{s:02d}_fold{test_b}.weights.h5"
            )
            ckpt = ModelCheckpoint(
                ckpt_path,
                monitor="val_loss",
                save_best_only=True,
                save_weights_only=True,
                mode="min"
            )

            model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=args.epochs,
                batch_size=args.batch_size,
                verbose=0,
                callbacks=[ckpt]
            )

            # evaluation best model
            model.load_weights(ckpt_path)
            loss, acc = model.evaluate(X_test, y_test, verbose=0)

            acc = float(acc)
            loss = float(loss)
            fold_acc.append(acc)

            rows.append({
                "subject": int(s),
                "fold": int(test_b + 1),
                "test_block": int(test_b),
                "val_block": int(val_b),
                "acc": acc,
                "loss": loss,
            })

        mean_s = float(np.mean(fold_acc))
        subject_means.append(mean_s)
        print(f"Subject {int(s):02d} mean_acc={mean_s:.4f}  folds={np.round(fold_acc,4)}")

    out_csv = os.path.join(args.outdir, "within_subject_4fold.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f, fieldnames=["subject", "fold", "test_block", "val_block", "acc", "loss"]
        )
        w.writeheader()
        w.writerows(rows)

    # moyenne et ecart type sur les sujets 
    mean_over_subjects = float(np.mean(subject_means))
    std_over_subjects = float(np.std(subject_means))

    print("\nWITHIN-SUBJECT 4-fold")
    print("Mean acc over subjects:", mean_over_subjects)
    print("Std  acc over subjects:", std_over_subjects)
    print(f"\nSaved: {out_csv}")
    print(f"Saved weights in: {args.outdir} (within_sXX_foldY.weights.h5)")


if __name__ == "__main__":
    main()
