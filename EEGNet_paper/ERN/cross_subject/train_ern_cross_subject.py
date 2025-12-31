import argparse, os, csv, random, sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from math import ceil

# EEGModels.py
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
EEGMODELS_DIR = os.path.abspath(os.path.join(THIS_DIR, ".", ".", "arl-eegmodels"))
sys.path.insert(0, EEGMODELS_DIR)
from EEGModels import EEGNet

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
    raise ValueError(
        f"Unexpected X shape {X.shape}. Expected (N,C,T) or (N,1,C,T) or (N,C,T,1)."
    )


def format_for_model(X_nct: np.ndarray, model_input_shape: tuple) -> np.ndarray:
    ish = tuple(model_input_shape)
    if len(ish) != 4:
        raise ValueError(f"Unsupported model input shape: {ish}")

    # channels_last: (None, C, T, 1)
    if ish[-1] == 1:
        return X_nct[..., np.newaxis].astype(np.float32)

    # channels_first: (None, 1, C, T)
    if ish[1] == 1:
        return X_nct[:, np.newaxis, :, :].astype(np.float32)

    raise ValueError(f"Unsupported model input shape: {ish}")


def compute_inverse_proportion_weights(y_int: np.ndarray):
    y_int = np.asarray(y_int).astype(int)
    classes, counts = np.unique(y_int, return_counts=True)
    if len(classes) != 2:
        raise ValueError(f"Expected binary labels for ERN, got classes={classes}")

    maj_i = int(np.argmax(counts))
    min_i = 1 - maj_i
    maj_c = int(classes[maj_i])
    min_c = int(classes[min_i])

    odds = counts[maj_i] / max(1, counts[min_i])
    w_min = int(ceil(odds))

    class_w = {maj_c: 1.0, min_c: float(w_min)}
    sample_w = np.vectorize(lambda yy: class_w[int(yy)])(y_int).astype(np.float32)
    return class_w, sample_w


def build_eegnet_82(nb_classes: int, Chans: int, Samples: int, dropout: float, kernLength: int):
    # EEGNet-8,2
    model = EEGNet(nb_classes=nb_classes, Chans=Chans, Samples=Samples, dropoutRate=dropout, kernLength=kernLength, F1=8, D=2, F2=16, dropoutType="Dropout",)
    
    model.compile(optimizer=Adam(), loss=CategoricalCrossentropy(), metrics=["accuracy"],)
    return model

def auc_binary(y_true_int: np.ndarray, y_proba_pos: np.ndarray) -> float:
    m = tf.keras.metrics.AUC(curve="ROC")
    m.update_state(y_true_int.astype(np.int32), y_proba_pos.astype(np.float32))
    return float(m.result().numpy())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_npz", required=True)
    ap.add_argument("--test_npz", required=True)
    ap.add_argument("--outdir", default="results_ern_cross_EEGNet82")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--reps", type=int, default=30)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    set_seeds(args.seed)
    rng = np.random.default_rng(args.seed)

    tr = np.load(args.train_npz, allow_pickle=True)
    te = np.load(args.test_npz, allow_pickle=True)

    # Train: X (N,C,T), y (N,), subject (N,)
    Xtr_nct = ensure_nct(tr["X"]).astype(np.float32)
    ytr_int = tr["y"].astype(np.int64)
    subj_tr = tr["subject"].astype(np.int64)

    # Test: X (N,C,T), subject (N,), optional y
    Xte_nct = ensure_nct(te["X"]).astype(np.float32)
    subj_te = te["subject"].astype(np.int64)
    yte_int = te["y"].astype(np.int64) if "y" in te.files else None

    nb_classes = 2
    Chans = int(Xtr_nct.shape[1])
    Samples = int(Xtr_nct.shape[2])

    train_subjects = np.unique(subj_tr)
    test_subjects = np.unique(subj_te)

    csv_path = os.path.join(args.outdir, "ern_cross_subject_folds.csv")
    write_header = not os.path.exists(csv_path)

    with open(csv_path, "a", newline="", encoding="utf-8") as fcsv:
        w = csv.writer(fcsv)
        if write_header:
            w.writerow([
                "rep", "train_subjects", "val_subjects",
                "n_train_trials", "n_val_trials",
                "class_weight", "val_loss", "val_acc", "val_auc",
                "test_auc_if_labels"
            ])

        for rep in range(args.reps):
            # Paper protocol: choose 4 validation subjects at random from TRAIN subjects.
            val_subs = rng.choice(train_subjects, size=4, replace=False)
            train_subs = np.array([s for s in train_subjects if s not in set(val_subs)],
                                  dtype=train_subjects.dtype)

            idx_train = np.where(np.isin(subj_tr, train_subs))[0]
            idx_val   = np.where(np.isin(subj_tr, val_subs))[0]

            X_train_nct = Xtr_nct[idx_train]
            y_train_int = ytr_int[idx_train]
            X_val_nct   = Xtr_nct[idx_val]
            y_val_int   = ytr_int[idx_val]

            class_w, sw_train = compute_inverse_proportion_weights(y_train_int)

            y_train = to_categorical(y_train_int, nb_classes)
            y_val   = to_categorical(y_val_int, nb_classes)

            tf.keras.backend.clear_session()

            # ERN : kernLength=64 (half of 128 Hz), dropout=0.25 for cross-subject
            model = build_eegnet_82(nb_classes, Chans, Samples, dropout=0.25, kernLength=64)

            # Match EEGModels.py input layout
            X_train = format_for_model(X_train_nct, model.input_shape)
            X_val   = format_for_model(X_val_nct, model.input_shape)
            X_test  = format_for_model(Xte_nct, model.input_shape)

            ckpt_path = os.path.join(args.outdir, f"ern_rep{rep:02d}.weights.h5")
            ckpt = ModelCheckpoint(
                ckpt_path,
                monitor="val_loss",
                save_best_only=True,
                save_weights_only=True,
                mode="min",
                verbose=0,
            )

            model.fit(
                X_train, y_train,
                sample_weight=sw_train,
                validation_data=(X_val, y_val),
                epochs=args.epochs,
                batch_size=args.batch_size,
                verbose=0,
                callbacks=[ckpt],
            )

            # keep weights at lowest validation loss
            model.load_weights(ckpt_path)

            val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
            val_proba_pos = model.predict(X_val, verbose=0)[:, 1]
            val_auc = auc_binary(y_val_int, val_proba_pos)

            # Predict test and save probabilities + metadata
            test_proba = model.predict(X_test, verbose=0).astype(np.float32)
            pred_path = os.path.join(args.outdir, f"ern_rep{rep:02d}_test_pred.npz")

            savez_kwargs = dict(proba=test_proba, subject=subj_te)
            for key in ["id_feedback", "session", "csv_file"]:
                if key in te.files:
                    savez_kwargs[key] = te[key]
            np.savez(pred_path, **savez_kwargs)

            test_auc = ""
            if yte_int is not None:
                test_auc = auc_binary(yte_int, test_proba[:, 1])

            w.writerow([
                rep,
                " ".join(map(str, train_subs.tolist())),
                " ".join(map(str, val_subs.tolist())),
                int(len(idx_train)),
                int(len(idx_val)),
                f"{class_w}",
                float(val_loss),
                float(val_acc),
                float(val_auc),
                test_auc
            ])
            fcsv.flush()

            print(f"[rep {rep+1:02d}/{args.reps}] val_auc={val_auc:.4f} saved={os.path.basename(ckpt_path)}")

    print(f"\nDone. Train subjects: {len(train_subjects)} | Test subjects (fixed): {len(test_subjects)}")
    print(f"Logs: {csv_path}")


if __name__ == "__main__":
    main()