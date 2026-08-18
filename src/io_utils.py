"""Figure-saving and cross-notebook cache helpers.

Every notebook in notebooks/00-11 is meant to be run as its own kernel, in
order -- there is no shared kernel state between them. Each notebook loads
whatever it needs from cache/ (written by earlier notebooks) and, if other
notebooks depend on what it computes, writes its own cache/<name>.pkl at the
end. Figures are saved to figures/ as PNGs whenever they're displayed.
"""
import os
import pickle

from src.config import CACHE_DIR, FIGURES_DIR


def get_device():
    import torch
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def save_fig(fig, name, subdir=None):
    """Save a matplotlib figure to figures/[subdir/]name.png."""
    d = FIGURES_DIR if subdir is None else os.path.join(FIGURES_DIR, subdir)
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, f'{name}.png')
    fig.savefig(path, dpi=200, bbox_inches='tight')
    return path


def save_cache(name, **objs):
    """Pickle keyword-arg variables to cache/<name>.pkl for later notebooks to load."""
    path = os.path.join(CACHE_DIR, f'{name}.pkl')
    with open(path, 'wb') as f:
        pickle.dump(objs, f)
    print(f'Cached {list(objs.keys())} -> {path}')
    return path


def load_cache(name):
    """Load cache/<name>.pkl (written by an earlier notebook) as a dict."""
    path = os.path.join(CACHE_DIR, f'{name}.pkl')
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found -- run notebooks/{name}.ipynb first (it writes this cache file).")
    with open(path, 'rb') as f:
        return pickle.load(f)
