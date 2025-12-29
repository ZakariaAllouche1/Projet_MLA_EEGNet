from __future__ import annotations

import argparse
import re
from fractions import Fraction
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
from scipy.signal import firwin, filtfilt, resample_poly


def _matlab_uint16_to_str(u16: np.ndarray) -> str:
    u16 = np.asarray(u16).reshape(-1)
    u16 = u16[u16 != 0]
    try:
        return "".join(chr(int(c)) for c in u16)
    except Exception:
        return ""


def _read_obj(f, obj) -> Any:
    import h5py

    if isinstance(obj, h5py.Dataset):
        data = obj[()]
        if isinstance(data, np.ndarray) and data.dtype == object:
            return data
        return data

    if isinstance(obj, h5py.Group):
        out = {}
        for k in obj.keys():
            out[k] = _read_obj(f, obj[k])
        return out

    return obj


def _deref(f, ref) -> Any:
    obj = f[ref]
    return _read_obj(f, obj)


def _read_cell_as_list(f, cell_ds) -> List[Any]:
    arr = cell_ds[()]
    arr = np.asarray(arr).reshape(-1)
    return [_deref(f, r) for r in arr]


def _read_labels_from_chanlocs(f, chanlocs_labels_ds) -> List[str]:
    arr = chanlocs_labels_ds[()]
    arr = np.asarray(arr).reshape(-1)
    labels: List[str] = []
    for r in arr:
        val = _deref(f, r)
        val = np.asarray(val)
        if val.dtype == np.uint16:
            labels.append(_matlab_uint16_to_str(val))
        elif val.dtype.kind in ("U", "S"):
            labels.append(str(val))
        else:
            labels.append(str(val.reshape(-1)[0]) if val.size else "")
    return labels


def _estimate_frac(fs_orig: float, fs_target: float) -> Fraction:
    return Fraction(fs_target / fs_orig).limit_denominator(1000)


def _fir_bandpass_ct(eeg_ct: np.ndarray, fs: float, l_freq: float, h_freq: float, transition_hz: float) -> np.ndarray:
    nyq = fs / 2.0
    if not (0 < l_freq < h_freq < nyq):
        raise ValueError(f"Bad band: l_freq={l_freq}, h_freq={h_freq}, nyq={nyq}")

    taps = int(round(3.3 * fs / max(transition_hz, 0.1)))
    taps = max(101, min(taps, 4001))
    if taps % 2 == 0:
        taps += 1

    b = firwin(numtaps=taps, cutoff=[l_freq, h_freq], pass_zero=False, fs=fs)
    return filtfilt(b, [1.0], eeg_ct, axis=-1).astype(np.float32, copy=False)


def _resample_ct(eeg_ct: np.ndarray, fs_orig: float, fs_target: float) -> Tuple[np.ndarray, Fraction]:
    if abs(fs_orig - fs_target) < 1e-9:
        return eeg_ct, Fraction(1, 1)
    frac = _estimate_frac(fs_orig, fs_target)
    out = resample_poly(eeg_ct, up=frac.numerator, down=frac.denominator, axis=-1).astype(np.float32, copy=False)
    return out, frac


def _onsets_from_markers_seq(seq: np.ndarray) -> np.ndarray:
    s = np.asarray(seq).reshape(-1)
    if s.size == 0:
        return np.zeros((0,), dtype=np.int64)
    prev = np.concatenate([[0], s[:-1]])
    on = np.flatnonzero((s != 0) & (s != prev))
    return on.astype(np.int64)


def _apply_car_continuous(eeg_ct: np.ndarray) -> np.ndarray:
    return (eeg_ct - eeg_ct.mean(axis=0, keepdims=True)).astype(np.float32, copy=False)


def _baseline_correct_epoch(ep_ct: np.ndarray, fs: float, tmin: float, bmin: float, bmax: float) -> np.ndarray:
    i0 = int(round((bmin - tmin) * fs))
    i1 = int(round((bmax - tmin) * fs))
    i0 = max(0, min(ep_ct.shape[1], i0))
    i1 = max(0, min(ep_ct.shape[1], i1))
    if i1 <= i0:
        return ep_ct
    base = ep_ct[:, i0:i1].mean(axis=1, keepdims=True)
    return (ep_ct - base).astype(np.float32, copy=False)


def _parse_subject_id(p: Path) -> int:
    m = re.search(r"s(\d+)", p.stem.lower())
    return int(m.group(1)) if m else -1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--which", choices=["train", "test", "all"], default="all")

    ap.add_argument("--fs-target", type=float, default=128.0)
    ap.add_argument("--tmin", type=float, default=0.0)
    ap.add_argument("--tmax", type=float, default=0.8)

    ap.add_argument("--demean", type=int, default=1)

    ap.add_argument("--do-filter", type=int, default=1)
    ap.add_argument("--l-freq", type=float, default=0.5)
    ap.add_argument("--h-freq", type=float, default=30.0)
    ap.add_argument("--fir-transition-hz", type=float, default=1.0)

    ap.add_argument("--do-car", type=int, default=1)
    ap.add_argument("--do-baseline", type=int, default=1)
    ap.add_argument("--baseline-tmin", type=float, default=-0.2)
    ap.add_argument("--baseline-tmax", type=float, default=0.0)

    ap.add_argument("--max-files", type=int, default=0)
    args = ap.parse_args()

    root = Path(args.root).resolve()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    mat_files = sorted(root.glob("s*.mat"))
    if args.max_files and args.max_files > 0:
        mat_files = mat_files[: args.max_files]
    if not mat_files:
        raise RuntimeError(f"No subject mats found in {root}")

    win_T = int(round((args.tmax - args.tmin) * args.fs_target))
    off_T = int(round(args.tmin * args.fs_target))
    if win_T <= 0:
        raise RuntimeError("Invalid window: tmax must be > tmin")

    print("== Setup ==")
    print("root:", root)
    print("outdir:", outdir)
    print("subjects:", len(mat_files))
    print("which:", args.which)
    print("fs_target:", args.fs_target, "| window:", (args.tmin, args.tmax), "| samples:", win_T)
    print("filter:", args.do_filter, "| band:", (args.l_freq, args.h_freq))
    print("CAR:", args.do_car, "| baseline:", args.do_baseline, "| baseline_win:", (args.baseline_tmin, args.baseline_tmax))
    print()

    buckets = ["train", "test"] if args.which == "all" else [args.which]
    bucket_data: Dict[str, Dict[str, List[Any]]] = {
        b: dict(X=[], y=[], subject=[], run=[], onset_sample=[], seq_code=[], src_file=[], fs_orig=[]) for b in buckets
    }

    ch_names_final: Optional[List[str]] = None
    fs_orig_nominal: Optional[float] = None

    import h5py

    warned_baseline = False

    for mp in mat_files:
        sid = _parse_subject_id(mp)
        with h5py.File(mp, "r") as f:
            ch_names = None
            if "RSVP" in f and isinstance(f["RSVP"], h5py.Group) and "chanlocs" in f["RSVP"]:
                cl = f["RSVP"]["chanlocs"]
                if isinstance(cl, h5py.Group) and "labels" in cl:
                    ch_names = _read_labels_from_chanlocs(f, cl["labels"])

            if ch_names is None:
                for key in ["train", "test"]:
                    if key in f and isinstance(f[key], h5py.Dataset):
                        runs = _read_cell_as_list(f, f[key])
                        if runs and isinstance(runs[0], dict) and "chanlocs" in runs[0]:
                            cl = runs[0]["chanlocs"]
                            if isinstance(cl, dict) and "labels" in cl:
                                labels_obj = cl["labels"]
                                labels_arr = np.asarray(labels_obj).reshape(-1)
                                tmp: List[str] = []
                                for r in labels_arr:
                                    val = _deref(f, r)
                                    val = np.asarray(val)
                                    tmp.append(_matlab_uint16_to_str(val) if val.dtype == np.uint16 else str(val))
                                ch_names = tmp
                                break
                    if ch_names is not None:
                        break

            if ch_names is None:
                raise RuntimeError(f"{mp.name}: cannot read channel labels.")

            if ch_names_final is None:
                ch_names_final = ch_names
            else:
                if len(ch_names) != len(ch_names_final):
                    raise RuntimeError(f"{mp.name}: channel count mismatch: {len(ch_names)} vs {len(ch_names_final)}")

            for bucket in buckets:
                if bucket not in f:
                    continue
                cell_ds = f[bucket]
                if not isinstance(cell_ds, h5py.Dataset):
                    continue

                runs = _read_cell_as_list(f, cell_ds)
                for ridx, runobj in enumerate(runs):
                    if not isinstance(runobj, dict):
                        continue
                    if "data" not in runobj or "markers_seq" not in runobj or "markers_target" not in runobj:
                        continue

                    data = np.asarray(runobj["data"], dtype=np.float32)
                    if data.ndim != 2:
                        continue
                    if data.shape[1] != len(ch_names):
                        if data.shape[0] == len(ch_names):
                            data = data.T
                        else:
                            raise RuntimeError(f"{mp.name}:{bucket}/run{ridx}: data shape {data.shape} unexpected.")

                    seq = np.asarray(runobj["markers_seq"]).reshape(-1)
                    tgt = np.asarray(runobj["markers_target"]).reshape(-1)
                    if seq.shape[0] != data.shape[0] or tgt.shape[0] != data.shape[0]:
                        raise RuntimeError(f"{mp.name}:{bucket}/run{ridx}: markers length mismatch with data.")

                    fs_orig = float(np.asarray(runobj.get("srate", 512.0)).reshape(-1)[0])
                    fs_orig_nominal = fs_orig_nominal or fs_orig

                    eeg = data.T.astype(np.float32, copy=False)

                    if args.demean:
                        eeg = eeg - eeg.mean(axis=1, keepdims=True)

                    if args.do_filter:
                        eeg = _fir_bandpass_ct(
                            eeg, fs=fs_orig, l_freq=args.l_freq, h_freq=args.h_freq, transition_hz=args.fir_transition_hz
                        )

                    eeg_ds, frac = _resample_ct(eeg, fs_orig=fs_orig, fs_target=args.fs_target)

                    if args.do_car:
                        eeg_ds = _apply_car_continuous(eeg_ds)

                    on = _onsets_from_markers_seq(seq)
                    if on.size == 0:
                        continue

                    seq_code = seq[on].astype(np.int64, copy=False)
                    tgt_code = tgt[on].astype(np.int64, copy=False)

                    keep_mask = (tgt_code == 1) | (tgt_code == 2)
                    on = on[keep_mask]
                    seq_code = seq_code[keep_mask]
                    tgt_code = tgt_code[keep_mask]
                    y = (tgt_code == 1).astype(np.int64)

                    on_ds = np.round(on.astype(np.float64) * (args.fs_target / fs_orig)).astype(np.int64)

                    kept = 0
                    for i in range(on_ds.size):
                        a = int(on_ds[i] + off_T)
                        b = a + win_T
                        if a < 0 or b > eeg_ds.shape[1]:
                            continue

                        ep = eeg_ds[:, a:b].astype(np.float32, copy=False)

                        if args.do_baseline:
                            if (not warned_baseline) and not (args.baseline_tmin < args.baseline_tmax):
                                print("WARN: baseline window invalid (baseline_tmin >= baseline_tmax)  baseline skipped.")
                                warned_baseline = True

                            ep2 = _baseline_correct_epoch(
                                ep, fs=args.fs_target, tmin=args.tmin, bmin=args.baseline_tmin, bmax=args.baseline_tmax
                            )
                            if ep2 is not ep:
                                ep = ep2
                            else:
                                if not warned_baseline:
                                    i0 = int(round((args.baseline_tmin - args.tmin) * args.fs_target))
                                    i1 = int(round((args.baseline_tmax - args.tmin) * args.fs_target))
                                    if i1 <= i0:
                                        print("WARN: baseline window not inside epoch baseline skipped.")
                                        warned_baseline = True

                        bd = bucket_data[bucket]
                        bd["X"].append(ep)
                        bd["y"].append(int(y[i]))
                        bd["subject"].append(int(sid))
                        bd["run"].append(int(ridx))
                        bd["onset_sample"].append(int(on[i]))
                        bd["seq_code"].append(int(seq_code[i]))
                        bd["src_file"].append(mp.name)
                        bd["fs_orig"].append(float(fs_orig))
                        kept += 1

                    print(
                        f"{mp.name} | {bucket}/run{ridx}: events={on_ds.size} kept={kept} "
                        f"fs={fs_orig:.1f}->{args.fs_target} ({frac.numerator}/{frac.denominator})"
                    )

    if ch_names_final is None:
        raise RuntimeError("No channels were read.")

    for bucket in buckets:
        bd = bucket_data[bucket]
        if not bd["X"]:
            print(f"\nWARN: bucket '{bucket}' produced 0 epochs.")
            continue

        X = np.stack(bd["X"], axis=0).astype(np.float32, copy=False)
        y = np.asarray(bd["y"], dtype=np.int64)

        payload = dict(
            X=X,
            y=y,
            ch_names=np.asarray(ch_names_final, dtype=object),
            subject=np.asarray(bd["subject"], dtype=np.int64),
            run=np.asarray(bd["run"], dtype=np.int64),
            onset_sample=np.asarray(bd["onset_sample"], dtype=np.int64),
            seq_code=np.asarray(bd["seq_code"], dtype=np.int64),
            src_file=np.asarray(bd["src_file"], dtype=object),
            fs=np.float32(args.fs_target),
            fs_orig_nominal=np.float32(fs_orig_nominal if fs_orig_nominal is not None else 512.0),
            fs_orig_est=np.asarray(bd["fs_orig"], dtype=np.float32),
            tmin=np.float32(args.tmin),
            tmax=np.float32(args.tmax),
            demean=np.int64(args.demean),
            filtered=np.int64(args.do_filter),
            l_freq=np.float32(args.l_freq),
            h_freq=np.float32(args.h_freq),
            fir_transition_hz=np.float32(args.fir_transition_hz),
            car=np.int64(args.do_car),
            baseline=np.int64(args.do_baseline),
            baseline_tmin=np.float32(args.baseline_tmin),
            baseline_tmax=np.float32(args.baseline_tmax),
            label_def=np.asarray("markers_target: 1=target, 2=nontarget", dtype=object),
        )

        out_path = outdir / f"p300_won2022_{bucket}_eegnet.npz"
        np.savez_compressed(out_path.as_posix(), **payload)

        tgt_frac = float((y == 1).mean())
        print("\n== Saved ==")
        print(out_path)
        print("X:", X.shape, "y:", y.shape, "target_frac:", tgt_frac)

    print("\nDone.")


if __name__ == "__main__":
    main()
