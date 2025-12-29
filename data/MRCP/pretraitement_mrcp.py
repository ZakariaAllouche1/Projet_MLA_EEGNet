import argparse
import glob
import os
import re
from typing import Any, Dict, Optional, Tuple

import numpy as np

CHANNELS = [
    "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T3", "C3", "Cz", "C4", "T4",
    "T5", "P3", "Pz", "P4", "T6", "O1", "O2", "Rt", "Lt"
]
CONDS = ["RightTap", "Rest", "LeftTap"]

FS_ORIG = 1024.0
TRIAL_SEC = 6.0
TRIAL_SAMPLES = int(FS_ORIG * TRIAL_SEC)
ONSET_SEC = 3.0


def safe_mkdir(p: str):
    os.makedirs(p, exist_ok=True)


def find_mats(root: str):
    mats = glob.glob(os.path.join(root, "Participant*.mat"))
    mats.sort()
    return mats


def _loadmat_scipy(path: str) -> Dict[str, Any]:
    from scipy.io import loadmat
    return loadmat(path)


def _load_cell_v73(path: str, var_hint: Optional[str] = None) -> np.ndarray:
    import h5py

    with h5py.File(path, "r") as f:
        var = None
        if var_hint and var_hint in f:
            var = var_hint
        else:
            for k in f.keys():
                if re.match(r"Participant\d+$", k):
                    var = k
                    break
            if var is None:
                var = list(f.keys())[0]

        cell_ds = f[var]
        out = np.empty(cell_ds.shape, dtype=object)
        it = np.nditer(np.zeros(cell_ds.shape, dtype=np.uint8), flags=["multi_index"])
        for _ in it:
            ref = cell_ds[it.multi_index]
            obj = f[ref]
            out[it.multi_index] = np.array(obj)
        return out


def load_participant_cell(mat_path: str) -> Tuple[int, np.ndarray]:
    base = os.path.basename(mat_path)
    m = re.match(r"Participant(\d+)\.mat$", base, re.IGNORECASE)
    if not m:
        raise RuntimeError(f"Nom de fichier inattendu: {base}")
    pid = int(m.group(1))

    try:
        d = _loadmat_scipy(mat_path)
        keys = [k for k in d.keys() if not k.startswith("__")]
        var = None
        for k in keys:
            if re.match(r"Participant\d+$", k):
                var = k
                break
        if var is None:
            var = keys[0]
        return pid, np.asarray(d[var])
    except Exception:
        return pid, _load_cell_v73(mat_path, var_hint=f"Participant{pid}")


def resample_poly_if_possible(x: np.ndarray, fs_in: float, fs_out: float) -> np.ndarray:
    if fs_in == fs_out:
        return x.astype(np.float32, copy=False)

    ratio = fs_in / fs_out
    if abs(ratio - round(ratio)) < 1e-9:
        up, down = 1, int(round(ratio))
    else:
        up, down = int(round(fs_out)), int(round(fs_in))

    try:
        from scipy.signal import resample_poly
        y = resample_poly(x, up=up, down=down, axis=-1)
        return y.astype(np.float32, copy=False)
    except Exception:
        t_in = np.arange(x.shape[-1]) / fs_in
        n_out = int(round(x.shape[-1] * fs_out / fs_in))
        t_out = np.arange(n_out) / fs_out
        y = np.empty((x.shape[0], n_out), dtype=np.float32)
        for c in range(x.shape[0]):
            y[c] = np.interp(t_out, t_in, x[c].astype(np.float64))
        return y


def bandpass_optional(x: np.ndarray, fs: float, l_freq: float, h_freq: float, method: str = "fir") -> np.ndarray:
    if l_freq <= 0 and h_freq <= 0:
        return x.astype(np.float32, copy=False)

    from scipy.signal import butter, filtfilt, firwin
    nyq = fs / 2.0

    def _iir():
        if l_freq <= 0:
            btype, Wn = "lowpass", h_freq / nyq
        elif h_freq <= 0:
            btype, Wn = "highpass", l_freq / nyq
        else:
            btype, Wn = "bandpass", [l_freq / nyq, h_freq / nyq]
        b, a = butter(4, Wn, btype=btype)
        return filtfilt(b, a, x, axis=-1).astype(np.float32, copy=False)

    if method.lower() == "iir":
        return _iir()

    n = x.shape[-1]
    max_taps = int(n // 3) - 1
    max_taps = max(31, max_taps)
    ntaps = min(max_taps, int(round(fs * 2)) | 1)
    if ntaps % 2 == 0:
        ntaps += 1
    ntaps = min(ntaps, max_taps if max_taps % 2 == 1 else max_taps - 1)

    try:
        if l_freq <= 0:
            taps = firwin(ntaps, h_freq / nyq, pass_zero="lowpass")
        elif h_freq <= 0:
            taps = firwin(ntaps, l_freq / nyq, pass_zero=False)
        else:
            taps = firwin(ntaps, [l_freq / nyq, h_freq / nyq], pass_zero=False)
        return filtfilt(taps, [1.0], x, axis=-1).astype(np.float32, copy=False)
    except Exception:
        return _iir()


def apply_baseline(epoch: np.ndarray, fs: float, tmin: float, tmax: float,
                   bmin: Optional[float], bmax: Optional[float]) -> np.ndarray:
    if bmin is None:
        bmin = tmin
    if bmax is None:
        bmax = 0.0

    bmin = max(bmin, tmin)
    bmax = min(bmax, tmax)

    s0 = int(round((bmin - tmin) * fs))
    s1 = int(round((bmax - tmin) * fs))

    if s1 <= s0 or s0 < 0 or s1 > epoch.shape[-1]:
        return epoch

    base = epoch[:, s0:s1].mean(axis=-1, keepdims=True)
    return (epoch - base).astype(np.float32, copy=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--fs-target", type=float, default=128.0)
    ap.add_argument("--tmin", type=float, default=-0.5)
    ap.add_argument("--tmax", type=float, default=1.0)
    ap.add_argument("--task", choices=["RL", "RLR", "MvR"], default="RL")

    ap.add_argument("--do-filter", type=int, default=1)
    ap.add_argument("--l-freq", type=float, default=0.1)
    ap.add_argument("--h-freq", type=float, default=40.0)
    ap.add_argument("--filter-method", choices=["fir", "iir"], default="fir")

    ap.add_argument("--baseline", type=int, default=1)
    ap.add_argument("--baseline-tmin", type=float, default=None)
    ap.add_argument("--baseline-tmax", type=float, default=0.0)

    ap.add_argument("--demean", type=int, default=0)
    ap.add_argument("--per-subject", type=int, default=1)
    ap.add_argument("--prefix", default="mrcp")
    args = ap.parse_args()

    root = os.path.abspath(args.root)
    outdir = os.path.abspath(args.outdir)
    safe_mkdir(outdir)

    mats = find_mats(root)
    if not mats:
        raise SystemExit("Aucun Participant*.mat trouvé")

    if args.task == "RL":
        cond_keep = [0, 2]
        label_map = {0: 0, 2: 1}
        class_names = ["RightTap", "LeftTap"]
    elif args.task == "RLR":
        cond_keep = [0, 1, 2]
        label_map = {0: 0, 1: 1, 2: 2}
        class_names = ["RightTap", "Rest", "LeftTap"]
    else:
        cond_keep = [0, 1, 2]
        label_map = {0: 1, 2: 1, 1: 0}
        class_names = ["Rest", "Movement"]

    eeg_ch_idx = list(range(19))
    ch_names = CHANNELS[:19]

    trial_len_out = int(round(TRIAL_SEC * args.fs_target))
    onset_out = int(round(ONSET_SEC * args.fs_target))
    s0_out = onset_out + int(round(args.tmin * args.fs_target))
    s1_out = onset_out + int(round(args.tmax * args.fs_target))
    if s0_out < 0 or s1_out > trial_len_out or s1_out <= s0_out:
        raise RuntimeError("Fenêtre invalide au fs cible")

    expected_T = s1_out - s0_out

    X_all, y_all = [], []
    subj_all, cond_all, trial_all, src_all = [], [], [], []

    for mp in mats:
        pid, cell = load_participant_cell(mp)
        if cell.shape[0] != 21 or cell.shape[1] != 3:
            raise RuntimeError(f"{mp}: shape inattendue {cell.shape}")

        n_trials = cell.shape[2]
        X_sub, y_sub = [], []
        subj_sub, cond_sub, trial_sub, src_sub = [], [], [], []

        for cond_i in cond_keep:
            for t in range(n_trials):
                trial_full = np.empty((len(eeg_ch_idx), TRIAL_SAMPLES), dtype=np.float32)
                ok = True
                for ci, ch in enumerate(eeg_ch_idx):
                    v = np.asarray(cell[ch, cond_i, t]).squeeze().reshape(-1)
                    if v.size != TRIAL_SAMPLES:
                        ok = False
                        break
                    trial_full[ci] = v.astype(np.float32, copy=False)
                if not ok:
                    continue

                if args.do_filter:
                    trial_full = bandpass_optional(
                        trial_full, fs=FS_ORIG, l_freq=args.l_freq, h_freq=args.h_freq, method=args.filter_method
                    )

                trial_rs = resample_poly_if_possible(trial_full, fs_in=FS_ORIG, fs_out=args.fs_target)

                if trial_rs.shape[-1] != trial_len_out:
                    if trial_rs.shape[-1] > trial_len_out:
                        trial_rs = trial_rs[:, :trial_len_out]
                    else:
                        pad = trial_len_out - trial_rs.shape[-1]
                        trial_rs = np.pad(trial_rs, ((0, 0), (0, pad)), mode="edge")

                epoch = trial_rs[:, s0_out:s1_out].astype(np.float32, copy=False)

                if args.baseline:
                    epoch = apply_baseline(
                        epoch, fs=args.fs_target, tmin=args.tmin, tmax=args.tmax,
                        bmin=args.baseline_tmin, bmax=args.baseline_tmax
                    )

                if args.demean:
                    epoch = (epoch - epoch.mean(axis=-1, keepdims=True)).astype(np.float32, copy=False)

                if epoch.shape[-1] != expected_T:
                    if epoch.shape[-1] > expected_T:
                        epoch = epoch[:, :expected_T]
                    else:
                        pad = expected_T - epoch.shape[-1]
                        epoch = np.pad(epoch, ((0, 0), (0, pad)), mode="edge")

                X_sub.append(epoch)
                y_sub.append(label_map[cond_i])
                subj_sub.append(pid)
                cond_sub.append(cond_i)
                trial_sub.append(t)
                src_sub.append(os.path.basename(mp))

        X_sub = np.stack(X_sub, axis=0) if X_sub else np.zeros((0, len(ch_names), expected_T), dtype=np.float32)
        y_sub = np.array(y_sub, dtype=np.int64) if y_sub else np.zeros((0,), dtype=np.int64)

        X_all.append(X_sub)
        y_all.append(y_sub)
        subj_all.append(np.array(subj_sub, dtype=np.int64))
        cond_all.append(np.array(cond_sub, dtype=np.int64))
        trial_all.append(np.array(trial_sub, dtype=np.int64))
        src_all.append(np.array(src_sub, dtype=object))

        if args.per_subject:
            outp = os.path.join(outdir, f"{args.prefix}_P{pid:02d}_eegnet.npz")
            np.savez_compressed(
                outp,
                X=X_sub,
                y=y_sub,
                subject=np.array(subj_sub, dtype=np.int64),
                condition=np.array(cond_sub, dtype=np.int64),
                trial=np.array(trial_sub, dtype=np.int64),
                src_file=np.array(src_sub, dtype=object),
                ch_names=np.array(ch_names, dtype=object),
                cond_names=np.array(CONDS, dtype=object),
                class_names=np.array(class_names, dtype=object),
                fs=args.fs_target,
                fs_orig=FS_ORIG,
                tmin=args.tmin,
                tmax=args.tmax,
                onset_sec=ONSET_SEC,
                dropped_channels=np.array(["Rt", "Lt"], dtype=object),
                filtered=args.do_filter,
                l_freq=args.l_freq,
                h_freq=args.h_freq,
                filter_method=args.filter_method,
                baseline=args.baseline,
                baseline_tmin=(args.baseline_tmin if args.baseline_tmin is not None else args.tmin),
                baseline_tmax=args.baseline_tmax,
                demean=args.demean,
            )

    X = np.concatenate(X_all, axis=0) if X_all else np.zeros((0, len(ch_names), expected_T), dtype=np.float32)
    y = np.concatenate(y_all, axis=0) if y_all else np.zeros((0,), dtype=np.int64)
    subject = np.concatenate(subj_all, axis=0) if subj_all else np.zeros((0,), dtype=np.int64)
    condition = np.concatenate(cond_all, axis=0) if cond_all else np.zeros((0,), dtype=np.int64)
    trial = np.concatenate(trial_all, axis=0) if trial_all else np.zeros((0,), dtype=np.int64)
    src_file = np.concatenate(src_all, axis=0) if src_all else np.zeros((0,), dtype=object)

    out_all = os.path.join(outdir, f"{args.prefix}_all_eegnet.npz")
    np.savez_compressed(
        out_all,
        X=X,
        y=y,
        subject=subject,
        condition=condition,
        trial=trial,
        src_file=src_file,
        ch_names=np.array(ch_names, dtype=object),
        cond_names=np.array(CONDS, dtype=object),
        class_names=np.array(class_names, dtype=object),
        fs=args.fs_target,
        fs_orig=FS_ORIG,
        tmin=args.tmin,
        tmax=args.tmax,
        onset_sec=ONSET_SEC,
        dropped_channels=np.array(["Rt", "Lt"], dtype=object),
        filtered=args.do_filter,
        l_freq=args.l_freq,
        h_freq=args.h_freq,
        filter_method=args.filter_method,
        baseline=args.baseline,
        baseline_tmin=(args.baseline_tmin if args.baseline_tmin is not None else args.tmin),
        baseline_tmax=args.baseline_tmax,
        demean=args.demean,
    )

    print(out_all)


if __name__ == "__main__":
    main()
