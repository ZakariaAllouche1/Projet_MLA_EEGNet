import argparse
import glob
import os
import re

import numpy as np


def find_gdf(root: str):
    files = glob.glob(os.path.join(root, "**", "A??[TE].gdf"), recursive=True)
    return sorted([f for f in files if os.path.isfile(f)])


def parse_subject_session(path: str):
    m = re.search(r"A(\d{2})([TE])\.gdf$", os.path.basename(path), re.IGNORECASE)
    if not m:
        return None, None
    return int(m.group(1)), m.group(2).upper()


def load_true_labels(mat_path: str) -> np.ndarray:
    from scipy.io import loadmat

    d = loadmat(mat_path)
    y = None
    for k, v in d.items():
        if k.startswith("__"):
            continue
        if isinstance(v, np.ndarray) and v.size == 288:
            y = v.reshape(-1)
            break

    if y is None:
        raise RuntimeError(f"Pas trouvé un vecteur 288 labels dans {mat_path}")

    y = y.astype(np.int64)
    if y.min() == 1 and y.max() == 4:
        y = y - 1
    return y


def butter_causal(data: np.ndarray, sfreq: float, l_freq: float = 4.0, h_freq: float = 40.0, order: int = 3):
    from scipy.signal import butter, lfilter

    nyq = 0.5 * sfreq
    b, a = butter(order, [l_freq / nyq, h_freq / nyq], btype="bandpass")
    return lfilter(b, a, data, axis=1)


def ema_standardize(data: np.ndarray, decay: float = 0.999, eps: float = 1e-4) -> np.ndarray:
    C, T = data.shape
    out = np.empty((C, T), dtype=np.float32)
    mean = np.zeros(C, dtype=np.float64)
    var = np.ones(C, dtype=np.float64)
    for t in range(T):
        x = data[:, t].astype(np.float64)
        mean = decay * mean + (1.0 - decay) * x
        diff = x - mean
        var = decay * var + (1.0 - decay) * (diff * diff)
        out[:, t] = (diff / np.sqrt(var + eps)).astype(np.float32)
    return out


def assign_trial_index(event_samples, trial_start_samples):
    ts = np.asarray(trial_start_samples)
    return np.searchsorted(ts, event_samples, side="right") - 1


def cat(xs):
    return np.concatenate(xs, axis=0) if xs else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out_train", default="smr_train_eegnet.npz")
    ap.add_argument("--out_test", default="smr_test_eegnet.npz")
    ap.add_argument("--drop_artifacts", type=int, default=1)
    args = ap.parse_args()

    import mne

    root = os.path.abspath(args.root)
    labels_dir = os.path.join(root, "true_labels")
    gdfs = find_gdf(root)
    if not gdfs:
        raise SystemExit("Aucun .gdf trouvé")

    fs = 128.0
    tmin = 0.5
    tmax = tmin + (256 - 1) / fs

    Xtr_list, ytr_list, subj_tr_list, trial_tr_list = [], [], [], []
    Xte_list, yte_list, subj_te_list, trial_te_list = [], [], [], []

    event_id = {str(k): k for k in (768, 769, 770, 771, 772, 783, 1023, 32766)}

    for gdf in gdfs:
        s, sess = parse_subject_session(gdf)
        if s is None:
            continue

        raw = mne.io.read_raw_gdf(gdf, preload=True, verbose="ERROR")
        picks = [i for i, ch in enumerate(raw.ch_names) if ch.upper().startswith("EEG")]
        raw.pick(picks)
        raw.resample(fs, npad="auto", verbose="ERROR")

        events, _ = mne.events_from_annotations(raw, event_id=event_id, verbose="ERROR")

        data = raw.get_data().astype(np.float32)
        data = butter_causal(data, sfreq=fs, l_freq=4.0, h_freq=40.0, order=3)
        data = ema_standardize(data, decay=0.999)
        raw._data[:] = data

        trial_starts = events[events[:, 2] == 768][:, 0]

        artifact_trials = set()
        if args.drop_artifacts:
            art = events[events[:, 2] == 1023][:, 0]
            if len(art) and len(trial_starts):
                idx = assign_trial_index(art, trial_starts)
                artifact_trials = set(int(i) for i in idx if i >= 0)

        if sess == "T":
            cue = events[np.isin(events[:, 2], [769, 770, 771, 772])]
            cue_trials = assign_trial_index(cue[:, 0], trial_starts)

            keep = cue_trials >= 0
            if args.drop_artifacts and artifact_trials:
                keep &= np.array([int(t) not in artifact_trials for t in cue_trials], dtype=bool)

            cue = cue[keep]
            cue_trials = cue_trials[keep]

            y = (cue[:, 2] - 769).astype(np.int64)

            epochs = mne.Epochs(raw, cue, event_id=None, tmin=tmin, tmax=tmax, baseline=None, preload=True, verbose="ERROR")
            X = epochs.get_data().astype(np.float32)

            Xtr_list.append(X)
            ytr_list.append(y)
            subj_tr_list.append(np.full(len(y), s, dtype=np.int64))
            trial_tr_list.append(cue_trials.astype(np.int64))

        else:
            cue = events[events[:, 2] == 783]
            cue_trials = assign_trial_index(cue[:, 0], trial_starts)

            keep = cue_trials >= 0
            if args.drop_artifacts and artifact_trials:
                keep &= np.array([int(t) not in artifact_trials for t in cue_trials], dtype=bool)

            cue = cue[keep]
            cue_trials = cue_trials[keep]

            y_all = load_true_labels(os.path.join(labels_dir, f"A{s:02d}E.mat"))
            y = y_all[cue_trials].astype(np.int64)

            epochs = mne.Epochs(raw, cue, event_id=None, tmin=tmin, tmax=tmax, baseline=None, preload=True, verbose="ERROR")
            X = epochs.get_data().astype(np.float32)

            Xte_list.append(X)
            yte_list.append(y)
            subj_te_list.append(np.full(len(y), s, dtype=np.int64))
            trial_te_list.append(cue_trials.astype(np.int64))

    np.savez(args.out_train, X=cat(Xtr_list), y=cat(ytr_list), subject=cat(subj_tr_list), trial=cat(trial_tr_list))
    np.savez(args.out_test, X=cat(Xte_list), y=cat(yte_list), subject=cat(subj_te_list), trial=cat(trial_te_list))


if __name__ == "__main__":
    main()
