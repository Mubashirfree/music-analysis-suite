from pathlib import Path

import numpy as np
import pandas as pd

from offline.config import DATA_FEATURES, DATA_PROCESSED


def load_manifest(kind: str) -> pd.DataFrame:
    """Load genre_manifest.csv or mood_manifest.csv; kind is 'genre' or 'mood'."""
    return pd.read_csv(DATA_PROCESSED / f"{kind}_manifest.csv")


def load_waveform(proc_path: str) -> np.ndarray:
    """Read a preprocessed song and convert it back to float32."""
    return np.load(proc_path).astype(np.float32)


def feature_path(feature: str, uid: str) -> Path:
    """Where the cached feature for one song lives."""
    folder = DATA_FEATURES / feature
    folder.mkdir(parents=True, exist_ok=True)
    return folder / f"{uid}.npy"


def is_cached(feature: str, uid: str) -> bool:
    return feature_path(feature, uid).exists()


def save_feature(feature: str, uid: str, array: np.ndarray) -> None:
    np.save(feature_path(feature, uid), array)


def load_feature(feature: str, uid: str) -> np.ndarray:
    return np.load(feature_path(feature, uid))