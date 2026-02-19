"""Save and load checkpoints to Google Drive."""

import pickle
from pathlib import Path

from utils.colab_setup import CHECKPOINT_DIR


def save_checkpoint(obj: object, name: str, epoch: int | None = None) -> Path:
    """Save a Python object to Drive as a pickle checkpoint."""
    suffix = f'_epoch{epoch}' if epoch is not None else ''
    path = CHECKPOINT_DIR / f'{name}{suffix}.pkl'
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(obj, f)
    print(f"Saved: {path}")
    return path


def load_checkpoint(name: str, epoch: int | None = None) -> object | None:
    """Load a checkpoint from Drive. Returns None if not found."""
    suffix = f'_epoch{epoch}' if epoch is not None else ''
    path = CHECKPOINT_DIR / f'{name}{suffix}.pkl'
    if path.exists():
        with open(path, 'rb') as f:
            return pickle.load(f)
    return None
