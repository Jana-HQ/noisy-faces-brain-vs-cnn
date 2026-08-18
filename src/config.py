"""Shared constants for the ecog_cnn_face_noise_analysis notebook series (notebooks/00-11).

Importing this module also creates data/, cache/, and figures/ if they don't exist yet.
"""
import os
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(REPO_ROOT, 'data')
CACHE_DIR = os.path.join(REPO_ROOT, 'cache')
FIGURES_DIR = os.path.join(REPO_ROOT, 'figures')
for _d in (DATA_DIR, CACHE_DIR, FIGURES_DIR):
    os.makedirs(_d, exist_ok=True)

RNG_SEED = 0

# subjects used throughout (others excluded for data-quality reasons upstream)
usable = [0, 3, 4, 5, 6]

# --- ECoG preprocessing windows ---
fs = 1000
trange_d1 = np.arange(-200, 400)             # dat1 epoch: valid 400ms ISI gap
baseline_d1 = (trange_d1 >= -200) & (trange_d1 < 0)
response_d1 = (trange_d1 >= 100) & (trange_d1 <= 350)

trange_d2 = np.arange(-200, 800)             # dat2 epoch: 1000ms stim, no ISI
response_win = (trange_d2 >= 100) & (trange_d2 <= 350)

# --- stimulus / condition grid ---
NOISE_LEVELS = np.arange(0, 101, 5)
CONDITIONS = [f'{cat}_{n}' for n in NOISE_LEVELS for cat in ['house', 'face']]
N_COND = len(CONDITIONS)
n_stim = 38  # matches Miller et al.'s stimulus-set size

# --- electrode groups ---
GROUP_NAMES = ['early_visual', 'ventral_temporal', 'face_selective', 'all']
MIN_TRIALS_PER_CONDITION = 8
MIN_GROUP_SIZE = 2
FACE_SEL_THRESH = 0.55   # face_selective group definition (dat2-derived, low-noise AUC)
FACE_SEL_CRIT = 0.60     # independent face-selectivity marker from the dat1 localizer

# --- sliding time window ---
WIN_WIDTH, WIN_STEP = 50, 25
WIN_STARTS = np.arange(0, 600 - WIN_WIDTH + 1, WIN_STEP)

# --- CNN layers ---
layer_names = ['pool1', 'pool2', 'pool3', 'pool4', 'pool5', 'fc6', 'fc7']

# --- collapse-threshold criterion ---
AUC_CRIT = 0.75   # fixed before looking at the neural result, from the CNN clean-curve knee
NOISE_HALFWIN = 10
POP_HALFWIN = 10
MIN_TRIALS_POP = 20
