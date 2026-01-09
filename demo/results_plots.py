from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import sem

import sys, inspect
import h5py
import tensorflow as tf
from tensorflow.keras import backend as K


def load_df(path: Path) -> pd.DataFrame:
    path = Path(path)
    suf = path.suffix.lower()

    if suf == ".csv":
        return pd.read_csv(path)

    if suf != ".json":
        raise ValueError(f"Format non supporté: {path}")

    try:
        df = pd.read_json(path)
        if isinstance(df, pd.Series):
            df = df.to_frame().T
        return df
    except Exception:
        pass

    try:
        df = pd.read_json(path, lines=True)
        if isinstance(df, pd.Series):
            df = df.to_frame().T
        return df
    except Exception:
        pass

    text = path.read_text(encoding="utf-8", errors="ignore")
    dec = json.JSONDecoder()

    objs = []
    i, n = 0, len(text)
    while i < n:
        while i < n and text[i].isspace():
            i += 1
        if i >= n:
            break
        obj, j = dec.raw_decode(text, i)
        objs.append(obj)
        i = j

    if not objs:
        raise ValueError(f"Aucun objet JSON détecté dans: {path}")

    if all(isinstance(o, dict) for o in objs):
        records = []
        list_keys = ["results", "records", "data", "folds", "runs"]
        for o in objs:
            found = False
            for k in list_keys:
                if k in o and isinstance(o[k], list) and (len(o[k]) == 0 or isinstance(o[k][0], dict)):
                    records.extend(o[k])
                    found = True
                    break
            if not found:
                records.append(o)
        return pd.json_normalize(records)

    if len(objs) == 1 and isinstance(objs[0], list):
        return pd.json_normalize(objs[0])

    return pd.json_normalize(objs)


def _metric_col(df: pd.DataFrame, metric: str) -> str:
    m = metric.lower()
    if m == "acc":
        prefs = [
            "acc", "test_acc", "val_acc", "accuracy",
            "metrics.accuracy", "metrics.acc", "metrics.balanced_accuracy",
        ]
    elif m == "auc":
        prefs = [
            "test_auc", "auc", "val_auc",
            "metrics.auc", "metrics.auc_ovr_macro", "metrics.auroc",
        ]
    else:
        prefs = [metric]

    for c in prefs:
        if c in df.columns:
            vals = pd.to_numeric(df[c], errors="coerce").values
            if np.isfinite(vals).any():
                return c

    raise KeyError(f"Pas de colonne pour metric='{metric}'. Colonnes: {list(df.columns)}")


def _two_sem(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size <= 1:
        return float("nan")
    return float(2 * sem(x))


def _to_subject_scalar(x):
    if isinstance(x, (list, tuple, np.ndarray)):
        return x[0] if len(x) > 0 else np.nan

    if isinstance(x, str):
        s = x.strip()
        if s.startswith("[") and s.endswith("]"):
            try:
                arr = json.loads(s)
                if isinstance(arr, list) and len(arr) > 0:
                    return arr[0]
            except Exception:
                pass
        if "," in s:
            return s.split(",")[0].strip()
        if " " in s:
            return s.split()[0].strip()
        return s

    return x


def _ensure_subject_col(df: pd.DataFrame) -> pd.DataFrame:
    if "subject" in df.columns:
        return df

    for cand in ["test_subject", "test_subjects", "test_subject_id"]:
        if cand in df.columns:
            df = df.copy()
            df["subject"] = df[cand].apply(_to_subject_scalar)
            return df

    raise KeyError(f"Pas de colonne sujet ('subject' ou 'test_subjects'). Colonnes: {list(df.columns)}")


def within_summary(path: Path, metric: str):
    df = load_df(path)
    df = _ensure_subject_col(df)

    col = _metric_col(df, metric)
    df = df.copy()
    df[col] = pd.to_numeric(df[col], errors="coerce")

    per_subject = df.groupby("subject")[col].mean().values
    per_subject = per_subject[np.isfinite(per_subject)]
    return float(np.mean(per_subject)), _two_sem(per_subject)


def cross_summary(path: Path | None, metric: str):
    if path is None:
        return float("nan"), float("nan")
    p = Path(path)
    if not p.exists():
        return float("nan"), float("nan")

    df = load_df(p)
    col = _metric_col(df, metric)
    x = pd.to_numeric(df[col], errors="coerce").values
    x = x[np.isfinite(x)]
    return float(np.mean(x)), _two_sem(x)


def plot_within_cross(name: str, metric: str,
                      authors_within: Path, authors_cross: Path,
                      ours_within: Path = None, ours_cross: Path = None,
                      authors_color=None, ours_color=None):

    a_w_mean, a_w_err = within_summary(authors_within, metric)
    a_c_mean, a_c_err = cross_summary(authors_cross, metric)

    o_w = None
    if ours_within is not None and Path(ours_within).exists():
        o_w = within_summary(ours_within, metric)

    o_c = None
    if ours_cross is not None and Path(ours_cross).exists():
        o_c = cross_summary(ours_cross, metric)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle(f"{name} Authors vs Ours")

    labels = ["Authors"]
    means = [a_w_mean]
    errs = [a_w_err]
    colors = [authors_color] if authors_color is not None else [None]

    if o_w is not None:
        labels += ["Ours"]
        means += [o_w[0]]
        errs += [o_w[1]]
        colors += [ours_color]

    x = np.arange(len(labels))
    if any(c is not None for c in colors):
        axes[0].bar(x, means, yerr=errs, capsize=6, color=colors)
    else:
        axes[0].bar(x, means, yerr=errs, capsize=6)

    axes[0].set_xticks(x, labels)
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel(metric.upper())
    axes[0].set_title("Within_subject")

    labels = ["Authors"]
    means = [a_c_mean]
    errs = [a_c_err]
    colors = [authors_color] if authors_color is not None else [None]

    if o_c is not None:
        labels += ["Ours"]
        means += [o_c[0]]
        errs += [o_c[1]]
        colors += [ours_color]

    x = np.arange(len(labels))
    if any(c is not None for c in colors):
        axes[1].bar(x, means, yerr=errs, capsize=6, color=colors)
    else:
        axes[1].bar(x, means, yerr=errs, capsize=6)

    axes[1].set_xticks(x, labels)
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel(metric.upper())
    axes[1].set_title("Cross_subject")

    plt.tight_layout()
    plt.show()



def plot_grouped_within_cross(items, metric="auc", title=None, ours_items=None,
                             authors_color=None, ours_color=None):
    names = [it[0] for it in items]
    n = len(items)

    aw_means, aw_errs, ac_means, ac_errs = [], [], [], []
    for _, w, c in items:
        m, e = within_summary(w, metric)
        aw_means.append(m); aw_errs.append(e)
        m, e = cross_summary(c, metric)
        ac_means.append(m); ac_errs.append(e)

    has_ours = ours_items is not None
    if has_ours:
        ow_means, ow_errs, oc_means, oc_errs = [], [], [], []
        for _, w, c in ours_items:
            m, e = within_summary(w, metric)
            ow_means.append(m); ow_errs.append(e)
            m, e = cross_summary(c, metric)
            oc_means.append(m); oc_errs.append(e)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(title or f"{metric.upper()} Authors vs Ours")

    x = np.arange(n)
    width = 0.35 if has_ours else 0.6

    if has_ours:
        kw_a = {"color": authors_color} if authors_color is not None else {}
        kw_o = {"color": ours_color} if ours_color is not None else {}

        axes[0].bar(x - width/2, aw_means, width, yerr=aw_errs, capsize=6, label="Authors", **kw_a)
        axes[0].bar(x + width/2, ow_means, width, yerr=ow_errs, capsize=6, label="Ours", **kw_o)
        axes[0].legend()
    else:
        kw_a = {"color": authors_color} if authors_color is not None else {}
        axes[0].bar(x, aw_means, width, yerr=aw_errs, capsize=6, label="Authors", **kw_a)

    axes[0].set_xticks(x, names)
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel(metric.upper())
    axes[0].set_title("Within_subject")

    ac_plot = np.nan_to_num(np.array(ac_means, float), nan=0.0)
    ae_plot = np.nan_to_num(np.array(ac_errs, float), nan=0.0)

    if has_ours:
        oc_plot = np.nan_to_num(np.array(oc_means, float), nan=0.0)
        oe_plot = np.nan_to_num(np.array(oc_errs, float), nan=0.0)

        kw_a = {"color": authors_color} if authors_color is not None else {}
        kw_o = {"color": ours_color} if ours_color is not None else {}

        axes[1].bar(x - width/2, ac_plot, width, yerr=ae_plot, capsize=6, label="Authors", **kw_a)
        axes[1].bar(x + width/2, oc_plot, width, yerr=oe_plot, capsize=6, label="Ours", **kw_o)
        axes[1].legend()
    else:
        kw_a = {"color": authors_color} if authors_color is not None else {}
        axes[1].bar(x, ac_plot, width, yerr=ae_plot, capsize=6, label="Authors", **kw_a)

    for i in range(n):
        if np.isnan(ac_means[i]) and (not has_ours or np.isnan(oc_means[i])):
            axes[1].text(i, 0.02, "NA", ha="center", va="bottom")

    axes[1].set_xticks(x, names)
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel(metric.upper())
    axes[1].set_title("Cross_subject")

    plt.tight_layout()
    plt.show()


def best_authors_fig7(subject: int, within_csv: Path, metric: str = "acc"):
    df = pd.read_csv(within_csv)
    df_s = df[df["subject"] == subject]
    best = df_s.sort_values(metric, ascending=False).iloc[0]

    best_fold = int(best["fold"])
    best_score = float(best[metric])
    weights_path = Path(within_csv).parent / f"within_s{subject:02d}_fold{best_fold}.weights.h5"
    return best_fold, best_score, weights_path

def extract_fig7_kernels_from_h5(weights_path: Path, chans=22, f1=8, d=2, klen=32):
    import h5py
    import numpy as np

    weights_path = Path(weights_path)

    def iter_datasets(g, prefix=""):
        for k in g.keys():
            obj = g[k]
            if isinstance(obj, h5py.Dataset):
                yield prefix + k, obj[()]
            elif isinstance(obj, h5py.Group):
                yield from iter_datasets(obj, prefix + k + "/")

    temporal = None
    spatial = None
    t_cands, s_cands = [], []

    with h5py.File(weights_path, "r") as f:
        for name, arr in iter_datasets(f):
            if not isinstance(arr, np.ndarray) or arr.ndim != 4:
                continue
            sh = tuple(arr.shape)

            if sorted(sh) == [1, 1, f1, klen] and (chans not in sh):
                t_cands.append((name, arr))

            if sorted(sh) == [1, d, f1, chans]:
                s_cands.append((name, arr))

    if not t_cands:
        raise RuntimeError("Temporal kernel introuvable dans le .h5 (shape attendue ~ [1,1,8,32]).")
    if not s_cands:
        raise RuntimeError("Spatial kernel introuvable dans le .h5 (shape attendue ~ [1,2,8,22]).")

    _, Wt = t_cands[0]
    _, Ws = s_cands[0]

    sh = Wt.shape
    if sh == (1, klen, 1, f1):                 # (1,32,1,8)
        temporal_kernels = np.stack([Wt[0, :, 0, i] for i in range(f1)], axis=0)
    elif sh == (f1, 1, 1, klen):               # (8,1,1,32)
        temporal_kernels = Wt[:, 0, 0, :]
    else:
        ax_k = sh.index(klen)
        ax_f = sh.index(f1)
        idx = [0, 0, 0, 0]
        temporal_kernels = np.zeros((f1, klen), dtype=Wt.dtype)
        for fi in range(f1):
            idx[ax_f] = fi
            for ki in range(klen):
                idx[ax_k] = ki
                temporal_kernels[fi, ki] = Wt[tuple(idx)]

    sh = Ws.shape
    if sh == (chans, 1, f1, d):
        spatial_filters = np.stack([[Ws[:, 0, fi, di] for di in range(d)] for fi in range(f1)], axis=0)
    else:
        ax_c = sh.index(chans)
        ax_f = sh.index(f1)
        ax_d = sh.index(d)
        spatial_filters = np.zeros((f1, d, chans), dtype=Ws.dtype)
        for fi in range(f1):
            for di in range(d):
                idx = [0, 0, 0, 0]
                idx[ax_f] = fi
                idx[ax_d] = di
                for ci in range(chans):
                    idx[ax_c] = ci
                    spatial_filters[fi, di, ci] = Ws[tuple(idx)]

    return temporal_kernels, spatial_filters

def extract_fig7_kernels_from_pt(
    ckpt_path: Path, *, chans=22, f1=8, d=2, klen=32,
    temporal_keys=("conv_time_short.weight", "conv_time_long.weight"),
    spatial_key="conv_spatial.weight",
):
    import numpy as np
    import torch
    from collections import OrderedDict

    ckpt_path = Path(ckpt_path)

    try:
        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
    except TypeError:
        ckpt = torch.load(str(ckpt_path), map_location="cpu")

    sd = ckpt
    if isinstance(ckpt, dict):
        for k in ["state_dict", "model_state_dict", "net", "model"]:
            if k in ckpt and isinstance(ckpt[k], (dict, OrderedDict)):
                sd = ckpt[k]
                break

    def to_np(t):
        return t.detach().cpu().numpy()

    temps = []
    for k in temporal_keys:
        if k not in sd:
            raise KeyError(f"Temporal key introuvable dans ckpt: {k}")
        W = to_np(sd[k])
        sh = W.shape

        if sh[-1] == klen and sh[1] == 1 and sh[2] == 1:
            temps.append(W[:, 0, 0, :])
        elif sh[2] == klen and sh[1] == 1 and sh[3] == 1:
            temps.append(W[:, 0, :, 0])
        else:
            raise ValueError(f"Shape temporal non gérée pour {k}: {sh}")

    temporal_kernels = np.concatenate(temps, axis=0)
    if temporal_kernels.shape != (f1, klen):
        raise ValueError(f"Temporal final inattendu: {temporal_kernels.shape} (attendu {(f1,klen)})")

    if spatial_key not in sd:
        raise KeyError(f"Spatial key introuvable dans ckpt: {spatial_key}")

    Ws = to_np(sd[spatial_key])
    shs = Ws.shape

    if shs == (f1 * d, 1, chans, 1):
        flat = Ws[:, 0, :, 0]
    elif shs == (f1 * d, 1, 1, chans):
        flat = Ws[:, 0, 0, :]
    else:
        raise ValueError(f"Shape spatial non gérée: {shs}")

    spatial_filters = flat.reshape(f1, d, chans)

    return temporal_kernels, spatial_filters



def plot_fig7(temporal_kernels, spatial_filters, title,
              sfreq=128.0, montage_name="standard_1020",
              cmap="jet", vlim_mode="row"):
    import numpy as np
    import matplotlib.pyplot as plt
    import mne

    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 10,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
    })

    temporal_kernels = np.asarray(temporal_kernels, float)
    spatial_filters  = np.asarray(spatial_filters, float)

    F1, Klen = temporal_kernels.shape
    F1b, D, Chans = spatial_filters.shape
    if F1b != F1:
        raise ValueError(f"F1 mismatch: temporal={F1} vs spatial={F1b}")

    t = np.arange(Klen) / float(sfreq)
    T = Klen / float(sfreq)

    ch_names = ["Fz","FC3","FC1","FCz","FC2","FC4",
                "C5","C3","C1","Cz","C2","C4","C6",
                "CP3","CP1","CPz","CP2","CP4",
                "P1","Pz","P2","POz"]
    if len(ch_names) != Chans:
        raise ValueError(f"Chans={Chans} mais len(ch_names)={len(ch_names)}")

    info = mne.create_info(ch_names=ch_names, sfreq=float(sfreq), ch_types=["eeg"] * len(ch_names))
    montage = mne.channels.make_standard_montage(montage_name)
    info.set_montage(montage, match_case=False)

    try:
        from mne.viz.topomap import _find_topomap_coords
    except Exception:
        from mne.channels.layout import _find_topomap_coords
    pos2d = _find_topomap_coords(info, picks=np.arange(len(ch_names)))

    if vlim_mode == "row":
        row_vmax = []
        for d_i in range(D):
            row_vmax.append(float(np.max(np.abs(spatial_filters[:, d_i, :])) + 1e-12))
    elif vlim_mode == "map":
        row_vmax = None
    else:
        raise ValueError("vlim_mode doit être 'row' ou 'map'")

    def topomap_article(ax, data, vmax=None):
        data = np.asarray(data, float)
        if vmax is None:
            vmax = float(np.max(np.abs(data)) + 1e-12)

        kwargs = dict(
            axes=ax, show=False,
            contours=0,
            cmap=cmap,
            res=128,
            image_interp="cubic",
            outlines="head",
            sensors=False,
        )

        if "extrapolate" in mne.viz.plot_topomap.__code__.co_varnames:
            kwargs["extrapolate"] = "local"

        try:
            im, _ = mne.viz.plot_topomap(data, info, vlim=(-vmax, vmax), **kwargs)
        except TypeError:
            im, _ = mne.viz.plot_topomap(data, info, vmin=-vmax, vmax=vmax, **kwargs)

        try:
            im.set_interpolation("bilinear")
        except Exception:
            pass

        ax.scatter(pos2d[:, 0], pos2d[:, 1], s=5, c="0.2")
        ax.set_axis_off()

    def scale_temporal(tkern):
        tk = np.asarray(tkern, float).copy()
        return tk / (np.max(np.abs(tk), axis=1, keepdims=True) + 1e-12) * 0.2

    temporal = scale_temporal(temporal_kernels)

    fig = plt.figure(figsize=(12, 4.2))
    gs = fig.add_gridspec(3, F1, height_ratios=[1.0, 1.55, 1.55], hspace=0.45, wspace=0.35)

    fig.text(0.1, 0.49, "Spat. Filter 1", rotation=90, va="center", ha="center", fontsize=10)
    fig.text(0.1, 0.18, "Spat. Filter 2", rotation=90, va="center", ha="center", fontsize=10)

    for f in range(F1):
        ax = fig.add_subplot(gs[0, f])
        ax.plot(t, temporal[f], linewidth=1.0)
        ax.axhline(0, linewidth=0.6)
        ax.set_xlim(0, T)
        ax.set_ylim(-0.22, 0.22)
        ax.set_xticks([0, T])
        ax.set_xticklabels(["0", f"{T:.2f}"])
        ax.set_yticks([-0.2, 0, 0.2] if f == 0 else [])
        ax.set_title(f"Temp. Filter {f+1}", fontsize=10)

        for d_i in range(D):
            ax2 = fig.add_subplot(gs[1 + d_i, f])
            vmax = row_vmax[d_i] if row_vmax is not None else None
            topomap_article(ax2, spatial_filters[f, d_i], vmax=vmax)

    fig.suptitle(title, y=0.98, fontsize=11)
    plt.show()



def _default_ch_names(C: int):
    if C == 19:
        return [
            "Fp1","Fp2","F7","F3","Fz","F4","F8",
            "T7","C3","Cz","C4","T8",
            "P7","P3","Pz","P4","P8","O1","O2"
        ]
    if C == 22:
        return ["Fz","FC3","FC1","FCz","FC2","FC4",
                "C5","C3","C1","Cz","C2","C4","C6",
                "CP3","CP1","CPz","CP2","CP4",
                "P1","Pz","P2","POz"]
    return [f"Ch{i+1}" for i in range(C)]


def _make_info(ch_names, sfreq=128.0, montage_name="standard_1020"):
    import mne
    info = mne.create_info(ch_names=ch_names, sfreq=float(sfreq), ch_types=["eeg"] * len(ch_names))
    try:
        montage = mne.channels.make_standard_montage(montage_name)
        info.set_montage(montage, match_case=False, on_missing="ignore")
    except Exception:
        pass
    return info


def _plot_topomap(ax, vec, info, cmap="jet", vlim=None, contours=6, sensors=False):
    import numpy as np
    import mne
    vec = np.asarray(vec, float)

    if vlim is None:
        vmax = float(np.nanmax(np.abs(vec)) + 1e-12)
        vlim = (-vmax, vmax)

    kwargs = dict(
        axes=ax, show=False,
        contours=contours,
        cmap=cmap,
        res=128,
        image_interp="cubic",
        outlines="head",
        sensors=sensors,
    )
    if "extrapolate" in mne.viz.plot_topomap.__code__.co_varnames:
        kwargs["extrapolate"] = "local"

    try:
        mne.viz.plot_topomap(vec, info, vlim=vlim, **kwargs)
    except TypeError:
        mne.viz.plot_topomap(vec, info, vmin=vlim[0], vmax=vlim[1], **kwargs)

    ax.set_axis_off()


def plot_fig9(
    panels,
    *,
    model_label="Authors",
    sfreq=128.0,
    t0=-0.5,
    tmax=None,
    topo_times=(0.05, 0.15),
    win=0.02,
    cmap="jet",
    ch_names=None,
    montage_name="standard_1020",
    vlim=None,
    figsize=(12.6, 4.6),
):
    import numpy as np
    import matplotlib.pyplot as plt

    if len(panels) != 3:
        raise ValueError("Figure 9 attend 3 panels (A,B,C).")

    R0 = np.asarray(panels[0]["R"], float)
    if R0.ndim != 2:
        raise ValueError(f"R doit être (C,T). Reçu: {R0.shape}")
    C, T = R0.shape

    for p in panels:
        R = np.asarray(p["R"], float)
        if R.shape != (C, T):
            raise ValueError(f"Incohérence shape: attendu {(C,T)} reçu {R.shape}")

    if ch_names is None:
        ch_names = _default_ch_names(C)
    if len(ch_names) != C:
        raise ValueError(f"len(ch_names)={len(ch_names)} mais C={C}")

    info = _make_info(ch_names, sfreq=sfreq, montage_name=montage_name)

    times = t0 + np.arange(T) / float(sfreq)

    t1 = t0 + T / float(sfreq) if tmax is None else float(tmax)

    if vlim is None:
        allv = np.concatenate([np.abs(np.asarray(p["R"], float)).ravel() for p in panels])
        vmax = float(np.nanpercentile(allv, 99) + 1e-12)
        vlim = (-vmax, vmax)

    def window_mean(R, tp):
        lo, hi = tp - win/2, tp + win/2
        idx = np.where((times >= lo) & (times <= hi))[0]
        if idx.size == 0:
            idx = np.array([int(np.argmin(np.abs(times - tp)))])
        return np.nanmean(R[:, idx], axis=1)

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 3, height_ratios=[1.35, 1.0], hspace=0.45, wspace=0.45)
    fig.suptitle(f"{model_label}", y=0.98, fontsize=12)

    for j, p in enumerate(panels):
        R = np.asarray(p["R"], float)

        ax = fig.add_subplot(gs[0, j])
        im = ax.imshow(
            R, aspect="auto", origin="lower",
            extent=[t0, t1, 0, C],
            vmin=vlim[0], vmax=vlim[1],
            cmap=cmap,
            interpolation="nearest",
        )
        ax.set_title(p.get("title", ""), fontsize=10)
        ax.set_xlabel("Time (seconds)")
        ax.set_ylabel("Channels")
        ax.text(-0.18, 1.02, p.get("letter",""), transform=ax.transAxes,
                fontsize=18, fontweight="bold", va="bottom")

        if np.isclose(t0, -0.5) and np.isclose(t1, 1.0):
            ax.set_xticks([-0.5, 0.0, 0.5, 1.0])

        cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        cb.set_ticks([vlim[0], 0.0, vlim[1]])

        sub = gs[1, j].subgridspec(1, 2, wspace=0.15)
        for k, tt in enumerate(topo_times):
            axi = fig.add_subplot(sub[0, k])
            vec = window_mean(R, tt)
            _plot_topomap(axi, vec, info, cmap=cmap, vlim=vlim, contours=6, sensors=False)
            axi.set_title(f"t \u2248 {tt:.4f}s", fontsize=9)

    plt.show()
    return fig


def integrated_gradients(model, x, target_idx: int, baseline=None, steps: int = 32):
    x = tf.convert_to_tensor(x, tf.float32)

    if baseline is None:
        baseline = tf.zeros_like(x)
    else:
        baseline = tf.convert_to_tensor(baseline, tf.float32)

    alphas = tf.linspace(0.0, 1.0, steps + 1)
    total_grads = tf.zeros_like(x)

    for a in alphas:
        xi = baseline + a * (x - baseline)
        with tf.GradientTape() as tape:
            tape.watch(xi)
            out = model(xi, training=False)
            score = out[:, target_idx]
        grads = tape.gradient(score, xi)
        total_grads += grads

    avg_grads = total_grads / tf.cast((steps + 1), tf.float32)
    ig = (x - baseline) * avg_grads
    return ig.numpy()



def infer_eegnet_params_from_weights(weights_path: Path, chans_guess=19):
    weights_path = Path(weights_path)

    def iter_arrays(f):
        def rec(g, prefix=""):
            for k in g.keys():
                obj = g[k]
                if isinstance(obj, h5py.Dataset):
                    arr = obj[()]
                    if isinstance(arr, np.ndarray):
                        yield prefix + k, arr
                elif isinstance(obj, h5py.Group):
                    yield from rec(obj, prefix + k + "/")
        yield from rec(f)

    temporal = None
    with h5py.File(weights_path, "r") as f:
        for name, arr in iter_arrays(f):
            if arr.ndim != 4:
                continue
            sh = tuple(arr.shape)
            ones = [d for d in sh if d == 1]
            if len(ones) == 2:
                others = [d for d in sh if d != 1]
                if len(others) == 2 and chans_guess not in others:
                    temporal = (name, sh, others)
                    break

    if temporal is None:
        raise RuntimeError("Impossible d'inférer kernLength/F1 depuis les poids (temporal conv introuvable).")

    _, sh, others = temporal
    kernLength = int(max(others))
    F1 = int(min(others))
    return dict(F1=F1, kernLength=kernLength)

def load_eegnet_official_from_weights(weights_path: Path, arl_dir: Path, chans=19, samples=192, nb_classes=2):
    arl_dir = Path(arl_dir)
    if str(arl_dir) not in sys.path:
        sys.path.append(str(arl_dir))

    from EEGModels import EEGNet

    p = infer_eegnet_params_from_weights(weights_path, chans_guess=chans)
    F1 = p["F1"]
    kernLength = p["kernLength"]

    base_kwargs = dict(
        nb_classes=nb_classes,
        Chans=chans,
        Samples=samples,
        dropoutRate=0.5,
        kernLength=kernLength,
        F1=F1,
        D=2,
        F2=16,
        norm_rate=0.25,
        dropoutType="Dropout",
    )

    sig = inspect.signature(EEGNet)
    filtered = {k: v for k, v in base_kwargs.items() if k in sig.parameters}

    last_err = None
    for fmt in ["channels_last", "channels_first"]:
        try:
            K.set_image_data_format(fmt)
            model = EEGNet(**filtered)
            model.load_weights(str(weights_path))
            return model, fmt, filtered
        except Exception as e:
            last_err = e

    raise RuntimeError(f"Impossible de charger les weights avec EEGNet officiel. Dernière erreur: {last_err}")


def integrated_gradients_torch(model, x, target_idx: int, *, steps: int = 32, baseline=None, device="cpu"):
    import torch

    model.eval()
    x = x.to(device).float()

    if baseline is None:
        baseline = torch.zeros_like(x)
    else:
        baseline = baseline.to(device).float()

    alphas = torch.linspace(0.0, 1.0, steps + 1, device=device).view(-1, *([1] * (x.ndim - 1)))
    x_interp = baseline + alphas * (x - baseline)
    x_interp.requires_grad_(True)

    out = model(x_interp)
    target = out[:, target_idx].sum()
    target.backward()

    grads = x_interp.grad

    grads = (grads[:-1] + grads[1:]) / 2.0
    avg_grads = grads.mean(dim=0, keepdim=True)

    ig = (x - baseline) * avg_grads
    return ig.detach().cpu().numpy()