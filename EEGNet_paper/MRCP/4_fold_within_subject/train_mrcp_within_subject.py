import argparse, os, sys, csv, glob, random
from math import ceil
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import to_categorical

# Utils
def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def ensure_nct(X):
    if X.ndim == 3:
        return X
    if X.ndim == 4 and X.shape[-1] == 1:
        return X[..., 0]
    if X.ndim == 4 and X.shape[1] == 1:
        return X[:, 0, :, :]
    raise ValueError("Unexpected X shape")

def format_for_model(X, input_shape):
    if input_shape[-1] == 1:
        return X[..., np.newaxis].astype(np.float32)
    if input_shape[1] == 1:
        return X[:, np.newaxis, :, :].astype(np.float32)
    raise ValueError("Unsupported model input shape")

def class_weight_paper(y):
    classes, counts = np.unique(y, return_counts=True)
    if len(classes) < 2:
        return None
    maj = counts.max()
    cw = {}
    for c, n in zip(classes, counts):
        cw[int(c)] = 1.0 if n == maj else float(ceil(maj / n))
    if all(v == 1.0 for v in cw.values()):
        return None
    return cw

def auc_roc(y_true, y_score):
    m = tf.keras.metrics.AUC(curve="ROC")
    m.update_state(y_true.astype(np.int32), y_score.astype(np.float32))
    return float(m.result().numpy())

# EEGNet-8,2
def build_eegnet_82(nb_classes, Chans, Samples, dropout, kernLength):
    from EEGModels import EEGNet
    model = EEGNet(nb_classes = nb_classes, Chans = Chans, Samples = Samples, dropoutRate = dropout, kernLength = kernLength, F1 = 8, D = 2, F2 = 16, dropoutType = "Dropout")
    model.compile(optimizer = Adam(), loss = CategoricalCrossentropy(), metrics = ["accuracy"])
    return model

# Stratified block builder
def make_stratified_blocks(y, n_blocks=4):
    blocks = [[] for _ in range(n_blocks)]
    for cls in np.unique(y):
        idx = np.where(y == cls)[0]
        splits = np.array_split(idx, n_blocks)
        for b in range(n_blocks):
            blocks[b].extend(splits[b])
    return [np.array(sorted(b)) for b in blocks]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required = True)
    ap.add_argument("--eegmodels_dir", required = True)
    ap.add_argument("--epochs", type = int, default = 500)
    ap.add_argument("--batch_size", type = int, default = 16)
    ap.add_argument("--seed", type = int, default = 42)
    ap.add_argument("--verbose", type = int, default = 0)
    ap.add_argument("--cpu_only", type = int, default = 0)
    args = ap.parse_args()

    if args.cpu_only == 1:
        tf.config.set_visible_devices([], "GPU")

    set_seeds(args.seed)
    sys.path.insert(0, os.path.abspath(args.eegmodels_dir))

    outdir = "results_mrcp_within_EEGNet82"
    os.makedirs(outdir, exist_ok = True)

    files = sorted(glob.glob(os.path.join(args.data_dir, "mrcp_P*_eegnet.npz")))
    if not files:
        raise RuntimeError("No subject NPZ files found")
    csv_path = os.path.join(outdir, "mrcp_within_subject_4fold_auc.csv")
    with open(csv_path, "w", newline="") as fcsv:
        w = csv.writer(fcsv)
        w.writerow(["subject", "fold", "n_train", "n_val", "n_test", "val_auc", "test_auc"])

        for path in files:
            d = np.load(path, allow_pickle=True)
            X = ensure_nct(d["X"]).astype(np.float32)
            y = d["y"].astype(np.int64)

            subj = int(d["subject"][0]) if "subject" in d else int(path.split("_")[1][1:])
            fs = float(d["fs"]) if "fs" in d else 128.0
            kernLength = int(fs / 2)

            blocks = make_stratified_blocks(y, 4)

            Chans, Samples = X.shape[1], X.shape[2]

            for test_fold in range(4):
                val_fold = (test_fold + 1) % 4
                train_folds = [i for i in range(4) if i not in (test_fold, val_fold)]

                idx_train = np.concatenate([blocks[i] for i in train_folds])
                idx_val = blocks[val_fold]
                idx_test = blocks[test_fold]

                Xtr, ytr = X[idx_train], y[idx_train]
                Xva, yva = X[idx_val],   y[idx_val]
                Xte, yte = X[idx_test],  y[idx_test]

                ytr_oh = to_categorical(ytr, 2)
                yva_oh = to_categorical(yva, 2)

                tf.keras.backend.clear_session()
                model = build_eegnet_82(2, Chans, Samples, dropout = 0.5, kernLength = kernLength)

                Xtr_f = format_for_model(Xtr, model.input_shape)
                Xva_f = format_for_model(Xva, model.input_shape)
                Xte_f = format_for_model(Xte, model.input_shape)

                weights = os.path.join(outdir, f"mrcp_S{subj:02d}_fold{test_fold}.weights.h5")
                ckpt = ModelCheckpoint(weights, monitor="val_loss",
                                       save_best_only=True, save_weights_only=True)

                model.fit(Xtr_f, ytr_oh, validation_data = (Xva_f, yva_oh), epochs = args.epochs, batch_size = args.batch_size, class_weight = class_weight_paper(ytr), callbacks = [ckpt], verbose = args.verbose)
                model.load_weights(weights)
                val_auc = auc_roc(yva, model.predict(Xva_f, verbose = 0)[:, 1])
                test_auc = auc_roc(yte, model.predict(Xte_f, verbose = 0)[:, 1])

                w.writerow([subj, test_fold,
                            len(idx_train), len(idx_val), len(idx_test),
                            f"{val_auc:.6f}", f"{test_auc:.6f}"])
                fcsv.flush()

                print(f"[MRCP within] S{subj:02d} fold{test_fold} "
                      f"val_auc={val_auc:.4f} test_auc={test_auc:.4f}")

    print("\nDONE — MRCP within-subject EEGNet-8,2")
    print(f"Results in: {outdir}")

if __name__ == "__main__":
    main()