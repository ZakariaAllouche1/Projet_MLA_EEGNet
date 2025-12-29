import argparse
import glob
import os
import re
from dataclasses import dataclass
from fractions import Fraction
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from scipy.signal import firwin, filtfilt, resample_poly

FNAME_RE = re.compile(r"Data_S(\d+)_Sess(\d+)\.csv$", re.IGNORECASE)
ID_RE = re.compile(r"^S(\d+)_Sess(\d+)_FB(\d+)$", re.IGNORECASE)

DEFAULT_ROOT = os.path.join("data", "ern_kaggle")


def parse_subject_session(path: str) -> Tuple[int, int]:
    m = FNAME_RE.search(os.path.basename(path))
    if not m:
        raise ValueError(f"Nom de fichier inattendu: {path}")
    return int(m.group(1)), int(m.group(2))


def parse_idfeedback(id_str: str) -> Optional[Tuple[int, int, int]]:
    m = ID_RE.match(str(id_str).strip())
    if not m:
        return None
    return int(m.group(1)), int(m.group(2)), int(m.group(3))


def estimate_fs_from_time(time_arr: np.ndarray) -> float:
    t = np.asarray(time_arr, dtype=float)
    dt = np.diff(t)
    dt = dt[np.isfinite(dt)]
    med = float(np.median(dt))
    return 1000.0 / med if med > 0.5 else 1.0 / med


def detect_onsets_from_event(ev: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    x = np.nan_to_num(ev.astype(float))
    above = x > threshold
    return (np.where(above[1:] & ~above[:-1])[0] + 1).astype(np.int64)


def bandpass_fir_zero_phase(
    x: np.ndarray, fs: float, low: float, high: float, transition_hz: float = 1.0
) -> np.ndarray:
    nyq = 0.5 * fs
    numtaps = int(round(3.3 * fs / max(transition_hz, 0.5)))
    if numtaps % 2 == 0:
        numtaps += 1
    b = firwin(numtaps, [low / nyq, high / nyq], pass_zero=False, window="hamming")
    return filtfilt(b, [1.0], x, axis=0).astype(np.float32)


def resample_continuous_to_fs(
    x: np.ndarray, fs_orig: float, fs_target: float, max_den: int = 1000
) -> Tuple[np.ndarray, int, int]:
    if abs(fs_target - fs_orig) < 1e-9:
        return x.astype(np.float32), 1, 1
    frac = Fraction(fs_target / fs_orig).limit_denominator(max_den)
    up, down = frac.numerator, frac.denominator
    y = resample_poly(x, up, down, axis=0)
    return y.astype(np.float32), up, down


@dataclass
class ExtractConfig:
    fs_target: float = 128.0
    tmin: float = 0.0
    tmax: float = 1.25
    do_filter: bool = False
    f_low: float = 1.0
    f_high: float = 40.0
    fir_transition_hz: float = 1.0
    event_threshold: float = 0.5
    pad_value: float = 0.0


def infer_eeg_cols_from_header(df: pd.DataFrame, time_col: str, event_col: str) -> List[str]:
    eeg_cols = [c for c in df.columns if c not in (time_col, event_col)]
    if "EOG" in eeg_cols and len(eeg_cols) == 57:
        eeg_cols.remove("EOG")
    return eeg_cols


def build_labels_index(labels_df: pd.DataFrame) -> Dict[Tuple[int, int], List[Tuple[int, int, str]]]:
    idx: Dict[Tuple[int, int], List[Tuple[int, int, str]]] = {}
    for id_str, pred in zip(labels_df["IdFeedBack"].tolist(), labels_df["Prediction"].tolist()):
        parsed = parse_idfeedback(id_str)
        if parsed is None:
            continue
        s, se, fb = parsed
        idx.setdefault((s, se), []).append((fb, int(pred), str(id_str)))
    for k in idx:
        idx[k].sort(key=lambda t: t[0])
    return idx


def extract_epochs_from_file(
    csv_path: str,
    eeg_cols: List[str],
    time_col: str,
    event_col: str,
    cfg: ExtractConfig,
    fs_orig: float,
) -> np.ndarray:
    usecols = [time_col, event_col] + eeg_cols
    df = pd.read_csv(csv_path, usecols=usecols)

    ev = pd.to_numeric(df[event_col], errors="coerce").fillna(0).to_numpy(np.float32)
    onsets = detect_onsets_from_event(ev, threshold=cfg.event_threshold)

    eeg = df[eeg_cols].to_numpy(np.float32)

    if cfg.do_filter:
        eeg = bandpass_fir_zero_phase(
            eeg, fs=fs_orig, low=cfg.f_low, high=cfg.f_high, transition_hz=cfg.fir_transition_hz
        )

    if abs(cfg.fs_target - fs_orig) > 1e-9:
        eeg_ds, up, down = resample_continuous_to_fs(eeg, fs_orig=fs_orig, fs_target=cfg.fs_target)
        onsets = np.round(onsets * (up / down)).astype(np.int64)
    else:
        eeg_ds = eeg.astype(np.float32)

    start_offset = int(round(cfg.tmin * cfg.fs_target))
    win_len = int(round((cfg.tmax - cfg.tmin) * cfg.fs_target))

    X = []
    n_time = eeg_ds.shape[0]
    n_ch = eeg_ds.shape[1]

    for o in onsets:
        s = int(o) + start_offset
        e = s + win_len
        if s < 0:
            continue
        if e <= n_time:
            seg = eeg_ds[s:e, :]
        else:
            seg = np.full((win_len, n_ch), cfg.pad_value, dtype=np.float32)
            avail = n_time - s
            if avail > 0:
                seg[:avail, :] = eeg_ds[s:, :]
        X.append(seg.T)

    if not X:
        return np.zeros((0, len(eeg_cols), win_len), dtype=np.float32)

    return np.stack(X, axis=0).astype(np.float32)


def make_idfeedback_strings(subject: int, session: int, n: int) -> np.ndarray:
    return np.array([f"S{subject:02d}_Sess{session:02d}_FB{i:03d}" for i in range(1, n + 1)], dtype=object)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--train-dir", default="train")
    ap.add_argument("--test-dir", default="test")
    ap.add_argument("--labels-csv", default="TrainLabels.csv")

    ap.add_argument("--time-col", default="Time")
    ap.add_argument("--event-col", default="FeedBackEvent")
    ap.add_argument("--fs-orig", default="auto")

    ap.add_argument("--fs-target", type=float, default=128.0)
    ap.add_argument("--tmin", type=float, default=0.0)
    ap.add_argument("--tmax", type=float, default=1.25)

    ap.add_argument("--do-filter", type=int, default=0)
    ap.add_argument("--f-low", type=float, default=1.0)
    ap.add_argument("--f-high", type=float, default=40.0)
    ap.add_argument("--fir-transition-hz", type=float, default=1.0)

    ap.add_argument("--event-threshold", type=float, default=0.5)

    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    train_path = os.path.join(args.root, args.train_dir)
    test_path = os.path.join(args.root, args.test_dir)
    labels_csv = os.path.join(args.root, args.labels_csv)

    train_files = sorted(glob.glob(os.path.join(train_path, "Data_S*_Sess*.csv")), key=parse_subject_session)
    test_files = sorted(glob.glob(os.path.join(test_path, "Data_S*_Sess*.csv")), key=parse_subject_session)

    df_sample = pd.read_csv(train_files[0], nrows=20000)
    eeg_cols = infer_eeg_cols_from_header(df_sample, args.time_col, args.event_col)

    fs_orig = estimate_fs_from_time(df_sample[args.time_col].to_numpy()) if args.fs_orig == "auto" else float(args.fs_orig)

    cfg = ExtractConfig(
        fs_target=float(args.fs_target),
        tmin=float(args.tmin),
        tmax=float(args.tmax),
        do_filter=bool(args.do_filter),
        f_low=float(args.f_low),
        f_high=float(args.f_high),
        fir_transition_hz=float(args.fir_transition_hz),
        event_threshold=float(args.event_threshold),
    )

    labels_df = pd.read_csv(labels_csv)
    labels_index = build_labels_index(labels_df)

    X_list, y_list, subj_list, sess_list, id_list, csv_list = [], [], [], [], [], []
    for f in train_files:
        s, se = parse_subject_session(f)
        Xf = extract_epochs_from_file(f, eeg_cols, args.time_col, args.event_col, cfg, fs_orig)

        entries = labels_index[(s, se)]
        if len(entries) != Xf.shape[0]:
            raise RuntimeError(f"Mismatch S{s:02d} Sess{se:02d}: epochs={Xf.shape[0]} vs labels={len(entries)}")

        y = np.array([pred for (_, pred, _) in entries], dtype=np.int64)
        ids = np.array([id_str for (_, _, id_str) in entries], dtype=object)

        X_list.append(Xf)
        y_list.append(y)
        subj_list.append(np.full((Xf.shape[0],), s, dtype=np.int64))
        sess_list.append(np.full((Xf.shape[0],), se, dtype=np.int64))
        id_list.append(ids)
        csv_list.append(np.full((Xf.shape[0],), os.path.basename(f), dtype=object))

    X_train = np.concatenate(X_list, axis=0).astype(np.float32)
    y_train = np.concatenate(y_list, axis=0).astype(np.int64)
    subject_train = np.concatenate(subj_list, axis=0)
    session_train = np.concatenate(sess_list, axis=0)
    id_feedback_train = np.concatenate(id_list, axis=0)
    csv_file_train = np.concatenate(csv_list, axis=0)

    Xt_list, subj_t_list, sess_t_list, id_t_list, csv_t_list = [], [], [], [], []
    for f in test_files:
        s, se = parse_subject_session(f)
        Xf = extract_epochs_from_file(f, eeg_cols, args.time_col, args.event_col, cfg, fs_orig)

        Xt_list.append(Xf)
        subj_t_list.append(np.full((Xf.shape[0],), s, dtype=np.int64))
        sess_t_list.append(np.full((Xf.shape[0],), se, dtype=np.int64))
        id_t_list.append(make_idfeedback_strings(s, se, Xf.shape[0]))
        csv_t_list.append(np.full((Xf.shape[0],), os.path.basename(f), dtype=object))

    X_test = np.concatenate(Xt_list, axis=0).astype(np.float32)
    subject_test = np.concatenate(subj_t_list, axis=0)
    session_test = np.concatenate(sess_t_list, axis=0)
    id_feedback_test = np.concatenate(id_t_list, axis=0)
    csv_file_test = np.concatenate(csv_t_list, axis=0)

    train_out = os.path.join(args.outdir, "ern_train_eegnet.npz")
    test_out = os.path.join(args.outdir, "ern_test_eegnet.npz")

    np.savez_compressed(
        train_out,
        X=X_train,
        y=y_train,
        subject=subject_train,
        session=session_train,
        id_feedback=id_feedback_train,
        csv_file=csv_file_train,
        fs=np.array(cfg.fs_target, dtype=np.float32),
        tmin=np.array(cfg.tmin, dtype=np.float32),
        tmax=np.array(cfg.tmax, dtype=np.float32),
        ch_names=np.array(eeg_cols, dtype=object),
        time_col=np.array(args.time_col, dtype=object),
        event_col=np.array(args.event_col, dtype=object),
        fs_orig=np.array(fs_orig, dtype=np.float32),
        filtered=np.array(int(cfg.do_filter), dtype=np.int64),
        f_low=np.array(cfg.f_low, dtype=np.float32),
        f_high=np.array(cfg.f_high, dtype=np.float32),
        fir_transition_hz=np.array(cfg.fir_transition_hz, dtype=np.float32),
        event_threshold=np.array(cfg.event_threshold, dtype=np.float32),
    )

    np.savez_compressed(
        test_out,
        X=X_test,
        subject=subject_test,
        session=session_test,
        id_feedback=id_feedback_test,
        csv_file=csv_file_test,
        fs=np.array(cfg.fs_target, dtype=np.float32),
        tmin=np.array(cfg.tmin, dtype=np.float32),
        tmax=np.array(cfg.tmax, dtype=np.float32),
        ch_names=np.array(eeg_cols, dtype=object),
        time_col=np.array(args.time_col, dtype=object),
        event_col=np.array(args.event_col, dtype=object),
        fs_orig=np.array(fs_orig, dtype=np.float32),
        filtered=np.array(int(cfg.do_filter), dtype=np.int64),
        f_low=np.array(cfg.f_low, dtype=np.float32),
        f_high=np.array(cfg.f_high, dtype=np.float32),
        fir_transition_hz=np.array(cfg.fir_transition_hz, dtype=np.float32),
        event_threshold=np.array(cfg.event_threshold, dtype=np.float32),
    )

    print(train_out)
    print(test_out)


if __name__ == "__main__":
    main()
