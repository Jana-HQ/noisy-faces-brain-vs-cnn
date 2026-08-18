"""Shared functions reused across the notebooks/00-11 analysis series.

Split into two groups:
  - context-free helpers (preprocessing, stats, RDMs, CNN hooks) that take all
    the data they need as arguments
  - pipeline builders (get_group_channels, build_ecog_group_rdm, elec_cache, ...)
    that are needed in more than one notebook, parametrized on the data objects
    each notebook loads from cache

Constants (fs, trange_d2, response_win, CONDITIONS, NOISE_LEVELS, ...) come from
src.config so call sites don't need to pass them explicitly.
"""
import os
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import signal, stats
from scipy.optimize import curve_fit
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from src import config as cfg

# ============================================================
# Data loading
# ============================================================

def load_alldat(data_dir=cfg.DATA_DIR):
    """Download (if needed) and load the Miller ECoG faces/houses dataset."""
    import requests
    fname = os.path.join(data_dir, 'faceshouses.npz')
    url = "https://osf.io/argh7/download"
    if not os.path.exists(fname):
        r = requests.get(url)
        with open(fname, "wb") as fid:
            fid.write(r.content)
    return np.load(fname, allow_pickle=True)['dat']


def load_stimuli(n_stim=cfg.n_stim, data_dir=cfg.DATA_DIR):
    """Load and resolution-match the face/house image set. Returns face_subset,
    house_subset, avg_mag_all (shared FFT magnitude spectrum), W, H."""
    import cv2
    import glob
    import subprocess
    from sklearn.datasets import fetch_lfw_people

    faces = fetch_lfw_people(color=False, resize=0.5).images
    face_subset = faces[:n_stim]
    H, W = face_subset.shape[1:]

    houses_repo = os.path.join(data_dir, 'Houses-dataset')
    if not os.path.exists(houses_repo):
        subprocess.run(['git', 'clone', '--depth', '1',
                         'https://github.com/emanhamed/Houses-dataset.git',
                         houses_repo], check=True)
    house_paths = sorted(glob.glob(os.path.join(houses_repo, 'Houses Dataset', '*_frontal.jpg')))[:n_stim]

    def load_house_matched(path, size=(W, H)):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
        return img.astype(np.float32) / 255.0

    house_subset = np.array([load_house_matched(p) for p in house_paths])
    avg_mag_all = compute_avg_magnitude(np.concatenate([face_subset, house_subset], axis=0))
    return face_subset, house_subset, avg_mag_all, W, H


# ============================================================
# ECoG preprocessing
# ============================================================

def common_average_reference(V, bad_channels=None):
    nchan = V.shape[1]
    bad_channels = bad_channels or []
    good = np.array([c for c in range(nchan) if c not in bad_channels])
    car = V[:, good].mean(axis=1, keepdims=True)
    return V[:, good] - car, good


def broadband_power(V, fs=1000):
    b, a = signal.butter(3, 50, btype='high', fs=fs)
    Vb = signal.filtfilt(b, a, V, axis=0)
    Vb = np.abs(Vb) ** 2
    b, a = signal.butter(3, 10, btype='low', fs=fs)
    Vb = signal.filtfilt(b, a, Vb, axis=0)
    return Vb / Vb.mean(axis=0)


def classify_region(gyrus):
    if not isinstance(gyrus, str):
        return 'other'
    g = gyrus.lower()
    if any(k in g for k in ['calcarine', 'occipital', 'lingual', 'cuneus']):
        return 'early_visual'
    if any(k in g for k in ['fusiform', 'parahippocampal', 'inferior temporal']):
        return 'ventral_temporal'
    return 'other'


# ============================================================
# Stimuli / phase scrambling
# ============================================================

def compute_avg_magnitude(imgs):
    return np.mean([np.abs(np.fft.fft2(im)) for im in imgs], axis=0)


def phase_scramble(img, noise_pct, avg_magnitude, seed=0):
    rng = np.random.RandomState(seed)
    F = np.fft.fft2(img)
    phase = np.angle(F)
    rand_phase = rng.uniform(-np.pi, np.pi, size=phase.shape)
    alpha = noise_pct / 100.0
    mixed_phase = (1 - alpha) * phase + alpha * rand_phase
    out = np.real(np.fft.ifft2(avg_magnitude * np.exp(1j * mixed_phase)))
    out -= out.min()
    out /= (out.max() + 1e-8)
    return out


# ============================================================
# Distance / classification statistics
# ============================================================

def crossnobis_1d(xa, xb, n_folds=4, seed=0):
    """Cross-validated distance between two 1D samples (single electrode)."""
    rng = np.random.RandomState(seed)
    xa, xb = np.asarray(xa), np.asarray(xb)
    if len(xa) < n_folds or len(xb) < n_folds:
        return np.nan
    fa = np.array_split(rng.permutation(len(xa)), n_folds)
    fb = np.array_split(rng.permutation(len(xb)), n_folds)
    d = []
    for k in range(n_folds):
        tra = np.concatenate([fa[i] for i in range(n_folds) if i != k])
        trb = np.concatenate([fb[i] for i in range(n_folds) if i != k])
        d.append((xa[tra].mean() - xb[trb].mean()) * (xa[fa[k]].mean() - xb[fb[k]].mean()))
    return float(np.mean(d))


def crossnobis_multivariate(X_a, X_b, n_folds=4, seed=0):
    """Cross-validated squared distance between two populations (electrodes or CNN units)."""
    rng = np.random.RandomState(seed)
    n_a, n_b = len(X_a), len(X_b)
    if n_a < n_folds or n_b < n_folds:
        return np.nan
    folds_a = np.array_split(rng.permutation(n_a), n_folds)
    folds_b = np.array_split(rng.permutation(n_b), n_folds)
    fold_dists = []
    for k in range(n_folds):
        test_a, test_b = folds_a[k], folds_b[k]
        train_a = np.concatenate([folds_a[i] for i in range(n_folds) if i != k])
        train_b = np.concatenate([folds_b[i] for i in range(n_folds) if i != k])
        diff_train = X_a[train_a].mean(0) - X_b[train_b].mean(0)
        diff_test = X_a[test_a].mean(0) - X_b[test_b].mean(0)
        fold_dists.append(diff_train @ diff_test)
    return float(np.mean(fold_dists))


def cv_auc(x, y, min_per_class=8):
    """Cross-validated AUC; StandardScaler lives inside the CV pipeline."""
    y = np.asarray(y).astype(int)
    if len(np.unique(y)) < 2:
        return np.nan
    counts = np.bincount(y)
    if counts.min() < min_per_class:
        return np.nan
    cv = StratifiedKFold(n_splits=min(5, counts.min()), shuffle=True, random_state=0)
    pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    return float(cross_val_score(pipe, np.asarray(x).reshape(-1, 1), y, cv=cv, scoring='roc_auc').mean())


def mean_ci95(arr, axis=0):
    """Mean and 95% CI half-width (normal approx.) along `axis`, NaN-safe."""
    arr = np.asarray(arr, dtype=float)
    n = np.sum(~np.isnan(arr), axis=axis)
    mean = np.nanmean(arr, axis=axis)
    sem = np.nanstd(arr, axis=axis, ddof=1) / np.sqrt(np.maximum(n, 1))
    ci95 = 1.96 * sem
    return mean, ci95


def upper_tri(M):
    return M[np.triu_indices(M.shape[0], k=1)]


def compare_rdms(rdm_a, rdm_b):
    """Spearman correlation between the upper triangles of two RDMs, NaN-safe."""
    if rdm_a is None or rdm_b is None:
        return np.nan, np.nan, 0
    a, b = upper_tri(rdm_a), upper_tri(rdm_b)
    mask = ~(np.isnan(a) | np.isnan(b))
    if mask.sum() < 10:
        return np.nan, np.nan, int(mask.sum())
    r, p = stats.spearmanr(a[mask], b[mask])
    return r, p, int(mask.sum())


def filter_outlier_trials(response, local_idx, trial_idx, z_thresh=3.0, min_keep=4):
    """Drop trials whose overall response magnitude (mean across this electrode
    group) is a |z|>z_thresh outlier. Falls back to the untouched trial set if
    that would leave too few trials -- a single extreme trial, combined with
    near-collinear electrodes in a small group, can dominate a crossnobis
    distance estimate (see notebooks/04 for the diagnostic that motivated this)."""
    if len(trial_idx) < min_keep:
        return trial_idx
    mag = response[trial_idx][:, local_idx].mean(axis=1)
    sd = mag.std()
    if sd == 0:
        return trial_idx
    z = (mag - mag.mean()) / sd
    keep = trial_idx[np.abs(z) <= z_thresh]
    return keep if len(keep) >= min_keep else trial_idx


def _z(v):
    v = np.asarray(v, float)
    s = v.std()
    return (v - v.mean()) / s if s > 0 else v * 0


def partial_spearman(a, b, ctrl):
    """Spearman(a,b) controlling for ctrl (here: noise level) -- the beyond-noise test."""
    m = np.isfinite(a) & np.isfinite(b) & np.isfinite(ctrl)
    if m.sum() < 15:
        return np.nan
    ra, rb, rc = _z(stats.rankdata(a[m])), _z(stats.rankdata(b[m])), _z(stats.rankdata(ctrl[m]))
    ea = ra - rc * (ra @ rc) / (rc @ rc)
    eb = rb - rc * (rb @ rc) / (rc @ rc)
    return float(np.corrcoef(ea, eb)[0, 1]) if ea.std() > 0 and eb.std() > 0 else np.nan


def threshold_cross(noise_grid, auc_curve, crit=cfg.AUC_CRIT):
    """Noise level where auc_curve first crosses below `crit` (linear interpolation)."""
    a = np.asarray(auc_curve, float)
    g = np.asarray(noise_grid, float)
    ok_ = np.isfinite(a)
    if ok_.sum() < 4:
        return np.nan
    g, a = g[ok_], a[ok_]
    if a[0] < crit:
        return np.nan
    below = np.where(a < crit)[0]
    if len(below) == 0:
        return float(g[-1])
    i = below[0]
    x0, x1, y0, y1 = g[i - 1], g[i], a[i - 1], a[i]
    return float(x0 + (crit - y0) * (x1 - x0) / (y1 - y0)) if y1 != y0 else float(g[i])


def corr_sigmoid(n, r0, theta, w):
    return r0 / (1 + np.exp((n - theta) / w))


def fit_corr_threshold(band_centers, r_curve, min_r_clean=0.05):
    m = np.isfinite(r_curve)
    if m.sum() < 4:
        return np.nan, np.nan, np.nan
    r0_guess = float(np.nanmax(r_curve))
    if r0_guess < min_r_clean:
        return np.nan, np.nan, np.nan
    try:
        popt, _ = curve_fit(corr_sigmoid, band_centers[m], r_curve[m],
                             p0=[max(r0_guess, 0.1), 50., 15.],
                             bounds=([0, 0, 2], [1, 100, 60]), maxfev=30000)
        r0, theta, w = popt
        pred = corr_sigmoid(band_centers[m], *popt)
        ss_r = float(((r_curve[m] - pred) ** 2).sum())
        ss_t = float(((r_curve[m] - r_curve[m].mean()) ** 2).sum())
        return float(theta), float(r0 / (4 * w)), (1 - ss_r / ss_t if ss_t > 0 else np.nan)
    except Exception:
        return np.nan, np.nan, np.nan


def auc_sigmoid(n, auc0, theta, w):
    return 0.5 + (auc0 - 0.5) / (1 + np.exp((n - theta) / w))


def fit_threshold(curve, grid=cfg.NOISE_LEVELS):
    m = np.isfinite(curve)
    if m.sum() < 8:
        return np.nan, np.nan, np.nan
    try:
        popt, _ = curve_fit(auc_sigmoid, grid[m], curve[m],
                             p0=[max(float(np.nanmax(curve)), .55), 50., 10.],
                             bounds=([.5, 0, 1], [1., 100, 60]), maxfev=30000)
        auc0, theta, w = popt
        pred = auc_sigmoid(grid[m], *popt)
        ss_r = float(((curve[m] - pred) ** 2).sum())
        ss_t = float(((curve[m] - curve[m].mean()) ** 2).sum())
        return float(theta), float((auc0 - .5) / (4 * w)), (1 - ss_r / ss_t if ss_t > 0 else np.nan)
    except Exception:
        return np.nan, np.nan, np.nan


# ============================================================
# RDM builders (ECoG and CNN share these)
# ============================================================

def build_layer_rdm_crossnobis(raw_acts_dict, layer_name, conditions=cfg.CONDITIONS, n_folds=4, seed=0):
    """One crossnobis RDM from a raw-activation cache: raw_acts_dict[layer_name][cond] -> (n_reps, n_units)."""
    n = len(conditions)
    rdm = np.full((n, n), np.nan)
    np.fill_diagonal(rdm, 0.0)
    for i, j in combinations(range(n), 2):
        Xi = raw_acts_dict[layer_name][conditions[i]]
        Xj = raw_acts_dict[layer_name][conditions[j]]
        rdm[i, j] = rdm[j, i] = crossnobis_multivariate(Xi, Xj, n_folds=n_folds, seed=seed)
    return rdm


def build_layer_rdms_bootstrap_once(raw_acts_dict, layer_names_list, conditions, n_trials_match, bootstrap_seed):
    """Resample n_trials_match reps (with replacement) per condition, then crossnobis -- for stability/CI checks."""
    rng = np.random.RandomState(bootstrap_seed)
    rdms = {}
    for L in layer_names_list:
        resampled = {c: raw_acts_dict[L][c][rng.randint(0, raw_acts_dict[L][c].shape[0], size=n_trials_match)]
                     for c in conditions}
        n = len(conditions)
        rdm = np.full((n, n), np.nan)
        np.fill_diagonal(rdm, 0.0)
        for i, j in combinations(range(n), 2):
            rdm[i, j] = rdm[j, i] = crossnobis_multivariate(resampled[conditions[i]], resampled[conditions[j]],
                                                              n_folds=4, seed=0)
        rdms[L] = rdm
    return rdms


def category_major_perm():
    """Index permutation that reorders CONDITIONS from noise-major (house_0,
    face_0, house_5, face_5, ...) to category-major (face block | house block),
    for RDM heatmap display."""
    conditions_by_category = [f'face_{n}' for n in cfg.NOISE_LEVELS] + [f'house_{n}' for n in cfg.NOISE_LEVELS]
    old_index = {c: i for i, c in enumerate(cfg.CONDITIONS)}
    return [old_index[c] for c in conditions_by_category]


def reorder_rdm(rdm, perm):
    return None if rdm is None else rdm[np.ix_(perm, perm)]


def auc_face_vs_house(raw_acts_dict, layer, noise_level):
    d = raw_acts_dict[layer]
    Xf, Xh = np.asarray(d[f'face_{noise_level}']), np.asarray(d[f'house_{noise_level}'])
    X = np.vstack([Xf, Xh])
    y = np.r_[np.ones(len(Xf)), np.zeros(len(Xh))]
    return cross_val_score(make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
                            X, y, cv=5, scoring='roc_auc').mean()


def get_noise_window_indices(center, width, noise_levels=cfg.NOISE_LEVELS, conditions=cfg.CONDITIONS):
    lo, hi = center - width / 2, center + width / 2
    levels_in_window = [n for n in noise_levels if lo <= n < hi]
    return [conditions.index(f'{cat}_{n}') for n in levels_in_window for cat in ['house', 'face']], levels_in_window


def subset_rdm(rdm, idx):
    return None if (rdm is None or len(idx) < 4) else rdm[np.ix_(idx, idx)]


_FACE0_IDX = cfg.CONDITIONS.index('face_0')
_HOUSE0_IDX = cfg.CONDITIONS.index('house_0')


def similarity_to_baseline(rdm, category):
    """-distance(clean, noise_n), same category -- i.e. how similar the noisy
    representation still is to the clean one, at each noise level."""
    base_idx = _FACE0_IDX if category == 'face' else _HOUSE0_IDX
    sims = np.full(len(cfg.NOISE_LEVELS), np.nan)
    for k, n in enumerate(cfg.NOISE_LEVELS):
        cond_idx = cfg.CONDITIONS.index(f'{category}_{n}')
        sims[k] = -rdm[base_idx, cond_idx]
    return sims


# ============================================================
# ECoG group-RDM pipeline (shared by notebooks 03, 04, 06, 09, 10)
# ============================================================

def get_group_channels(electrode_type_df, s, group_name, auc_thresh=cfg.FACE_SEL_THRESH):
    sub = electrode_type_df[electrode_type_df['subject'] == s]
    if group_name == 'face_selective':
        chans = sub.loc[sub['auc_face_low_noise'] > auc_thresh, 'channel'].tolist()
    elif group_name == 'ventral_temporal':
        chans = sub.loc[sub['region'] == 'ventral_temporal', 'channel'].tolist()
    elif group_name == 'early_visual':
        chans = sub.loc[sub['region'] == 'early_visual', 'channel'].tolist()
    elif group_name == 'all':
        chans = sub['channel'].tolist()
    else:
        raise ValueError(group_name)
    return sorted(int(c) for c in chans)


def get_subject_response(alldat, s):
    """Per-subject dat2 response matrix (trials x electrodes) + condition trial indices."""
    d2 = alldat[s][1]
    V_raw = d2['V'].astype('float64')
    t_on = d2['t_on']
    cat = d2['stim_cat'].ravel()
    noise = d2['stim_noise'].ravel()
    V_car, good_chans = common_average_reference(V_raw, bad_channels=None)
    BB = broadband_power(V_car, fs=cfg.fs)
    ts = t_on[:, np.newaxis] + cfg.trange_d2
    valid = (ts.min(axis=1) >= 0) & (ts.max(axis=1) < BB.shape[0])
    ts, cat_v, noise_v = ts[valid], cat[valid], noise[valid]
    BB_epochs = BB[ts, :]
    response = BB_epochs[:, cfg.response_win, :].mean(axis=1)
    response = (response - response.mean(0)) / (response.std(0) + 1e-9)
    cond_idx = {}
    for n in cfg.NOISE_LEVELS:
        cond_idx[f'house_{n}'] = np.where((noise_v == n) & (cat_v == 1))[0]
        cond_idx[f'face_{n}'] = np.where((noise_v == n) & (cat_v == 2))[0]
    return response, good_chans, cond_idx


def build_ecog_group_rdm(alldat, electrode_type_df, s, group_name, n_folds=4, seed=0,
                          reject_outliers=True, z_thresh=3.0):
    response, good_chans, cond_idx = get_subject_response(alldat, s)
    group_chans = [c for c in get_group_channels(electrode_type_df, s, group_name) if c in good_chans]
    if len(group_chans) < cfg.MIN_GROUP_SIZE:
        return None, group_chans
    local_idx = [np.where(good_chans == c)[0][0] for c in group_chans]
    X = response[:, local_idx]
    rdm = np.full((cfg.N_COND, cfg.N_COND), np.nan)
    np.fill_diagonal(rdm, 0.0)
    for i, j in combinations(range(cfg.N_COND), 2):
        idx_i, idx_j = cond_idx[cfg.CONDITIONS[i]], cond_idx[cfg.CONDITIONS[j]]
        if reject_outliers:
            idx_i = filter_outlier_trials(response, local_idx, idx_i, z_thresh=z_thresh)
            idx_j = filter_outlier_trials(response, local_idx, idx_j, z_thresh=z_thresh)
        if len(idx_i) < cfg.MIN_TRIALS_PER_CONDITION or len(idx_j) < cfg.MIN_TRIALS_PER_CONDITION:
            continue
        d = crossnobis_multivariate(X[idx_i], X[idx_j], n_folds=n_folds, seed=seed)
        rdm[i, j] = rdm[j, i] = d
    return rdm, group_chans


def build_subject_epoch_cache(alldat):
    """Cache raw dat2 epochs once per subject -- reused by every sliding-window computation."""
    cache = {}
    for s in cfg.usable:
        d2 = alldat[s][1]
        V_raw = d2['V'].astype('float64')
        t_on = d2['t_on']
        cat = d2['stim_cat'].ravel()
        noise = d2['stim_noise'].ravel()
        V_car, good_chans = common_average_reference(V_raw, bad_channels=None)
        BB = broadband_power(V_car, fs=cfg.fs)
        ts = t_on[:, np.newaxis] + cfg.trange_d2
        valid = (ts.min(axis=1) >= 0) & (ts.max(axis=1) < BB.shape[0])
        ts, cat_v, noise_v = ts[valid], cat[valid], noise[valid]
        BB_epochs = BB[ts, :]
        cond_idx = {}
        for n in cfg.NOISE_LEVELS:
            cond_idx[f'house_{n}'] = np.where((noise_v == n) & (cat_v == 1))[0]
            cond_idx[f'face_{n}'] = np.where((noise_v == n) & (cat_v == 2))[0]
        cache[s] = dict(BB_epochs=BB_epochs, good_chans=good_chans, cond_idx=cond_idx)
    return cache


def build_group_rdm_windowed(subject_epoch_cache, electrode_type_df, s, group_name, win_lo_ms, win_hi_ms,
                              n_folds=4, seed=0, reject_outliers=True, z_thresh=3.0):
    cache = subject_epoch_cache[s]
    BB_epochs, good_chans, cond_idx = cache['BB_epochs'], cache['good_chans'], cache['cond_idx']
    group_chans = [c for c in get_group_channels(electrode_type_df, s, group_name) if c in good_chans]
    if len(group_chans) < cfg.MIN_GROUP_SIZE:
        return None
    win_mask = (cfg.trange_d2 >= win_lo_ms) & (cfg.trange_d2 < win_hi_ms)
    local_idx = [np.where(good_chans == c)[0][0] for c in group_chans]
    response = BB_epochs[:, win_mask, :][:, :, local_idx].mean(axis=1)
    all_cols = np.arange(response.shape[1])
    rdm = np.full((cfg.N_COND, cfg.N_COND), np.nan)
    np.fill_diagonal(rdm, 0.0)
    for i, j in combinations(range(cfg.N_COND), 2):
        idx_i, idx_j = cond_idx[cfg.CONDITIONS[i]], cond_idx[cfg.CONDITIONS[j]]
        if reject_outliers:
            idx_i = filter_outlier_trials(response, all_cols, idx_i, z_thresh=z_thresh)
            idx_j = filter_outlier_trials(response, all_cols, idx_j, z_thresh=z_thresh)
        if len(idx_i) < cfg.MIN_TRIALS_PER_CONDITION or len(idx_j) < cfg.MIN_TRIALS_PER_CONDITION:
            continue
        d = crossnobis_multivariate(response[idx_i], response[idx_j], n_folds=n_folds, seed=seed)
        rdm[i, j] = rdm[j, i] = d
    return rdm


# ============================================================
# Per-electrode pipeline (shared by notebooks 07, 08, 09)
# ============================================================

_COND_NOISE = np.array([int(c.split('_')[1]) for c in cfg.CONDITIONS])
_COND_ISFACE = np.array([c.startswith('face') for c in cfg.CONDITIONS]).astype(int)


def build_elec_cache(alldat):
    """Per-electrode dat2 responses (z-scored) + raw epochs, keyed by subject."""
    cache = {}
    for s in cfg.usable:
        d2 = alldat[s][1]
        V_car, good = common_average_reference(d2['V'].astype('float64'))
        BB = broadband_power(V_car, fs=cfg.fs)
        ts = d2['t_on'][:, np.newaxis] + cfg.trange_d2
        valid = (ts.min(1) >= 0) & (ts.max(1) < BB.shape[0])
        ep = BB[ts[valid], :]
        resp = ep[:, cfg.response_win, :].mean(1)
        resp = (resp - resp.mean(0)) / (resp.std(0) + 1e-9)
        cache[s] = dict(resp=resp, good=good,
                         cat=np.asarray(d2['stim_cat']).ravel()[valid],
                         noise=np.asarray(d2['stim_noise']).ravel()[valid],
                         epochs=ep)
    return cache


def electrode_rdm(elec_cache, s, ch):
    c = elec_cache[s]
    if ch not in c['good']:
        return None
    loc = np.where(c['good'] == ch)[0][0]
    x = c['resp'][:, loc]
    idx = {i: np.where((c['noise'] == _COND_NOISE[i]) & (c['cat'] == (2 if _COND_ISFACE[i] else 1)))[0]
           for i in range(cfg.N_COND)}
    R = np.full((cfg.N_COND, cfg.N_COND), np.nan)
    np.fill_diagonal(R, 0.0)
    for i, j in combinations(range(cfg.N_COND), 2):
        a, b = idx[i], idx[j]
        if len(a) >= 4 and len(b) >= 4:
            R[i, j] = R[j, i] = crossnobis_1d(x[a], x[b])
    return R


# ============================================================
# CNN hooks
# ============================================================

def get_hook(name, is_conv, act_dict):
    def hook(module, inp, out):
        act_dict[name] = out.mean(dim=[2, 3]).detach() if is_conv else out.detach()
    return hook


def register_vgg_hooks(model, act_dict):
    """Register the 5 pool + 2 fc forward hooks used throughout (ImageNet-VGG16 and VGGFace)."""
    model.features[4].register_forward_hook(get_hook('pool1', True, act_dict))
    model.features[9].register_forward_hook(get_hook('pool2', True, act_dict))
    model.features[16].register_forward_hook(get_hook('pool3', True, act_dict))
    model.features[23].register_forward_hook(get_hook('pool4', True, act_dict))
    model.features[30].register_forward_hook(get_hook('pool5', True, act_dict))
    model.classifier[0].register_forward_hook(get_hook('fc6', False, act_dict))
    model.classifier[3].register_forward_hook(get_hook('fc7', False, act_dict))


def make_noise_hook(name, is_conv, act_dict, noise_r, noise_mode):
    """noise_r: dict with key 'value' (float). noise_mode: dict with key 'mode'
    ('pool1_only' or 'all_layers'). Passed as mutable dicts so the noise level
    can be changed between forward passes without re-registering hooks."""
    import torch

    def hook(module, inp, out):
        r = noise_r['value']
        apply_here = (noise_mode['mode'] == 'all_layers') or \
                     (noise_mode['mode'] == 'pool1_only' and name == 'pool1')
        if r > 0 and apply_here:
            std = torch.sqrt(r * torch.abs(out) + 1e-12)
            out_noisy = out + torch.randn_like(out) * std
        else:
            out_noisy = out
        act_dict[name] = out_noisy.mean(dim=[2, 3]).detach() if is_conv else out_noisy.detach()
        return out_noisy
    return hook


def register_noise_hooks(model, act_dict, noise_r, noise_mode):
    """Like register_vgg_hooks, but every layer's output can be perturbed by
    noise_r/noise_mode (mutable dicts) between forward passes. Clears any
    hooks already on the module (e.g. from register_vgg_hooks) first."""
    hooked = {
        'pool1': (model.features[4], True), 'pool2': (model.features[9], True),
        'pool3': (model.features[16], True), 'pool4': (model.features[23], True),
        'pool5': (model.features[30], True), 'fc6': (model.classifier[0], False),
        'fc7': (model.classifier[3], False),
    }
    for name, (module, is_conv) in hooked.items():
        module._forward_hooks.clear()
        module.register_forward_hook(make_noise_hook(name, is_conv, act_dict, noise_r, noise_mode))


# ============================================================
# CNN forward-pass utilities (shared by notebooks 05, 10, 11)
# ============================================================

def get_cnn_preprocess():
    import torchvision.transforms as T
    return T.Compose([
        T.ToPILImage(), T.Resize((224, 224)), T.Grayscale(num_output_channels=3),
        T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def make_cnn_tensor(img, noise_pct, avg_magnitude, seed, preprocess):
    scrambled = phase_scramble(img, noise_pct, avg_magnitude=avg_magnitude, seed=seed)
    return preprocess(scrambled.astype(np.float32))


def run_cnn_batch(model, act_dict, names, tensor_list, device, batch_size=128):
    import torch
    all_acts = {name: [] for name in names}
    with torch.no_grad():
        for i in range(0, len(tensor_list), batch_size):
            batch = torch.stack(tensor_list[i:i + batch_size]).to(device)
            model(batch)
            for name in names:
                all_acts[name].append(act_dict[name].cpu().numpy())
    return {name: np.concatenate(v, axis=0) for name, v in all_acts.items()}


def cache_raw_cnn_activations(model, act_dict, names, face_subset, house_subset, avg_mag_all,
                               preprocess, device, n_seeds=3, batch_size=128):
    """Cache raw per-image activations for every (category, noise level) condition."""
    raw_acts = {name: {} for name in names}
    for cat_name, images in [('face', face_subset), ('house', house_subset)]:
        for n in cfg.NOISE_LEVELS:
            tensor_list = []
            for img_idx, img in enumerate(images):
                for seed_i in range(n_seeds):
                    tensor_list.append(make_cnn_tensor(img, n, avg_mag_all, img_idx * 1000 + seed_i, preprocess))
            acts = run_cnn_batch(model, act_dict, names, tensor_list, device, batch_size=batch_size)
            for name in names:
                raw_acts[name][f'{cat_name}_{n}'] = acts[name]
        print(f"{cat_name}: all {len(cfg.NOISE_LEVELS)} noise levels cached")
    return raw_acts


def run_cnn_repeats(model, act_dict, names, noise_pct, avg_magnitude, images, n_repeats, base_seed, rng,
                     preprocess, device):
    tensor_list = []
    for k in range(n_repeats):
        img_idx = rng.randint(0, len(images))
        tensor_list.append(make_cnn_tensor(images[img_idx], noise_pct, avg_magnitude, base_seed + k, preprocess))
    return run_cnn_batch(model, act_dict, names, tensor_list, device, batch_size=n_repeats)


def build_cnn_layer_rdms_for_r(r_value, model, act_dict, names, face_subset, house_subset, avg_mag_all,
                                noise_r, preprocess, device, n_repeats=15, seed=0,
                                min_trials_per_condition_cnn=8):
    """One crossnobis RDM per layer at internal-noise ratio r_value (repeat-based:
    each condition is re-sampled n_repeats times at random images/phase seeds,
    since a fixed 38-image cache can't capture repeat-to-repeat noise variance)."""
    noise_r['value'] = r_value
    rng = np.random.RandomState(seed)
    cond_repeats = {}
    for cat_name, images in [('face', face_subset), ('house', house_subset)]:
        cat_offset = 0 if cat_name == 'face' else 100_000
        for n in cfg.NOISE_LEVELS:
            base_seed = cat_offset + int(n) * 100 + int(r_value * 10)
            cond_repeats[f'{cat_name}_{n}'] = run_cnn_repeats(model, act_dict, names, n, avg_mag_all,
                                                                images, n_repeats, base_seed, rng, preprocess, device)
    rdms = {}
    for L in names:
        rdm = np.full((cfg.N_COND, cfg.N_COND), np.nan)
        np.fill_diagonal(rdm, 0.0)
        for i, j in combinations(range(cfg.N_COND), 2):
            Xi, Xj = cond_repeats[cfg.CONDITIONS[i]][L], cond_repeats[cfg.CONDITIONS[j]][L]
            if Xi.shape[0] >= min_trials_per_condition_cnn and Xj.shape[0] >= min_trials_per_condition_cnn:
                rdm[i, j] = rdm[j, i] = crossnobis_multivariate(Xi, Xj, n_folds=4, seed=0)
        rdms[L] = rdm
    return rdms


def compare_noise_sweep_to_ecog(rdms_by_r, mode_label, ecog_group_rdms, layer_names_list, r_values,
                                 group_names=cfg.GROUP_NAMES, usable=cfg.usable):
    rows = []
    for r in r_values:
        for s in usable:
            for group_name in group_names:
                ecog_rdm = ecog_group_rdms.get((s, group_name))
                for L in layer_names_list:
                    corr, p, n = compare_rdms(ecog_rdm, rdms_by_r[r][L])
                    rows.append(dict(r=r, subject=s, group=group_name, layer=L, corr=corr, p=p, n_pairs=n,
                                      noise_mode=mode_label))
    df = pd.DataFrame(rows)
    df['layer'] = pd.Categorical(df['layer'], categories=layer_names_list, ordered=True)
    return df


def build_comparison_df(ecog_rdms, cnn_rdms, layer_names_list, group_names=cfg.GROUP_NAMES, usable=cfg.usable):
    rows = []
    for s in usable:
        for group_name in group_names:
            ecog_rdm = ecog_rdms.get((s, group_name))
            for L in layer_names_list:
                r, p, n = compare_rdms(ecog_rdm, cnn_rdms[L])
                rows.append(dict(subject=s, group=group_name, layer=L, r=r, p=p, n_pairs=n))
    df = pd.DataFrame(rows)
    df['layer'] = pd.Categorical(df['layer'], categories=layer_names_list, ordered=True)
    return df
