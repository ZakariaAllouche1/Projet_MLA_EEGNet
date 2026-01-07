#!/usr/bin/env python3
"""
EEGNet-8,2 — P300 — CROSS-SUBJECT (FORCED GPU)

Protocol (Lawhern et al., 2018):
- Fixed TRAIN subjects
- Fixed TEST subjects
- 30 repetitions
- 4 validation subjects randomly sampled from TRAIN
- 500 epochs FIXED per repeat (NO EarlyStopping)
- Metrics computed at END of each repeat
- Epoch-wise train/val loss logged to .txt
- Repeat-wise metrics logged to .csv
"""

# ======================= FORCE GPU =======================
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# ======================= IMPORTS =======================
import sys
import csv
import random
import numpy as np
import tensorflow as tf
from datetime import datetime

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.utils import to_categorical

from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
)

# ======================= GPU CHECK =======================
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    print(f"✔ GPU detected: {gpus}")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
else:
    print("❌ NO GPU DETECTED — RUNNING ON CPU")

# ======================= PATH TO EEGModels =======================
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
EEGMODELS_DIR = os.path.abspath(os.path.join(THIS_DIR, ".."))
sys.path.insert(0, EEGMODELS_DIR)

from EEGModels import EEGNet   # OFFICIAL EEGNet

# ======================= CONFIG =======================
SEED = 42
N_REPEATS = 30
EPOCHS = 500
BATCH_SIZE = 16

# EEGNet-8,2 (paper)
F1, D, F2 = 8, 2, 16
KERN_LENGTH = 64
DROPOUT = 0.25

# ======================= DATA PATHS =======================
TRAIN_PATH = "../data/p300_npz/p300_won2022_train_eegnet.npz"
TEST_PATH  = "../data/p300_npz/p300_won2022_test_eegnet.npz"

# ======================= LOG PATHS =======================
LOG_DIR = "/home/mla/MLA_EEGNet/logs/p300_cross_abbas"
os.makedirs(LOG_DIR, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_CSV = os.path.join(LOG_DIR, f"p300_cross_eegnet82_{timestamp}.csv")
LOG_TXT = os.path.join(LOG_DIR, f"p300_cross_eegnet82_{timestamp}.txt")

# ======================= SEEDS =======================
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ======================= LOAD DATA =======================
train = np.load(TRAIN_PATH, allow_pickle=True)
test  = np.load(TEST_PATH, allow_pickle=True)

X_train = train["X"].astype("float32")
y_train = train["y"].astype(int)
subjects_train = train["subject"].astype(int)

X_test = test["X"].astype("float32")
y_test = test["y"].astype(int)

if X_train.ndim == 3:
    X_train = X_train[..., None]
    X_test = X_test[..., None]

CHANS = X_train.shape[1]
SAMPLES = X_train.shape[2]

print("\n================ DATA =================")
print(f"Train: {X_train.shape} | subjects={len(np.unique(subjects_train))}")
print(f"Test : {X_test.shape}")

# ======================= MODEL =======================
def build_model():
    model = EEGNet(
        nb_classes=2,
        Chans=CHANS,
        Samples=SAMPLES,
        dropoutRate=DROPOUT,
        kernLength=KERN_LENGTH,
        F1=F1,
        D=D,
        F2=F2,
        dropoutType="Dropout",
    )
    model.compile(
        optimizer=Adam(),
        loss=CategoricalCrossentropy(),
        metrics=["accuracy"],
    )
    return model

# ======================= METRICS =======================
def compute_metrics(y_true, y_prob):
    y_pred = (y_prob >= 0.5).astype(int)
    return {
        "AUC": roc_auc_score(y_true, y_prob),
        "Accuracy": accuracy_score(y_true, y_pred),
        "BalancedAcc": balanced_accuracy_score(y_true, y_pred),
        "F1": f1_score(y_true, y_pred),
    }

# ======================= EPOCH LOGGER =======================
class EpochLogger(Callback):
    def __init__(self, logfile):
        self.logfile = logfile
        self.best = np.inf

    def on_epoch_end(self, epoch, logs=None):
        tr = logs["loss"]
        va = logs["val_loss"]
        mark = ""
        if va < self.best:
            self.best = va
            mark = " *best*"
        with open(self.logfile, "a") as f:
            f.write(
                f"Epoch {epoch+1:03d} | "
                f"tr_loss={tr:.4f} "
                f"va_loss={va:.4f}{mark}\n"
            )

# ======================= CSV HEADER =======================
with open(LOG_CSV, "w", newline="") as f:
    csv.writer(f).writerow([
        "repeat",
        "epochs_trained",
        "test_auc",
        "test_accuracy",
        "balanced_accuracy",
        "f1_score",
    ])

# ======================= TXT HEADER =======================
with open(LOG_TXT, "w") as f:
    f.write("EEGNet-8,2 — P300 — CROSS-SUBJECT\n\n")

# ======================= TRAINING =======================
unique_subjects = np.unique(subjects_train)

for rep in range(N_REPEATS):

    print(f"\n================ REPEAT {rep+1}/{N_REPEATS} ================")

    with open(LOG_TXT, "a") as f:
        f.write(f"\n================ REPEAT {rep+1}/{N_REPEATS} ================\n")

    # ---- subject split ----
    val_subjects = np.random.choice(unique_subjects, 4, replace=False)
    train_subjects = [s for s in unique_subjects if s not in val_subjects]

    idx_tr = np.isin(subjects_train, train_subjects)
    idx_va = np.isin(subjects_train, val_subjects)

    X_tr = X_train[idx_tr]
    X_va = X_train[idx_va]
    Y_tr = to_categorical(y_train[idx_tr], 2)
    Y_va = to_categorical(y_train[idx_va], 2)

    tf.keras.backend.clear_session()
    model = build_model()

    logger = EpochLogger(LOG_TXT)

    model.fit(
        X_tr, Y_tr,
        validation_data=(X_va, Y_va),
        epochs=EPOCHS,          
        batch_size=BATCH_SIZE,
        callbacks=[logger],
        verbose=0,
    )

    # ---- TEST ----
    y_prob = model.predict(X_test, verbose=0)[:, 1]
    m = compute_metrics(y_test, y_prob)

    # ---- PRINT METRICS ----
    print(
        f"[REPEAT {rep+1}/{N_REPEATS}] "
        f"AUC={m['AUC']:.4f} | "
        f"Acc={m['Accuracy']:.4f} | "
        f"BalAcc={m['BalancedAcc']:.4f} | "
        f"F1={m['F1']:.4f}"
    )

    # ---- CSV LOG ----
    with open(LOG_CSV, "a", newline="") as f:
        csv.writer(f).writerow([
            rep + 1,
            EPOCHS,
            m["AUC"],
            m["Accuracy"],
            m["BalancedAcc"],
            m["F1"],
        ])

print("\n================ FINISHED ================")
print("CSV:", LOG_CSV)
print("TXT:", LOG_TXT)
