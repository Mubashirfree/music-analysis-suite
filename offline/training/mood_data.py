import numpy as np

from offline.data.mood import MOOD_CLASSES
from offline.features.cache import load_feature, load_manifest

TARGETS = ["valence_mean", "arousal_mean"]


def pooled_embedding(uid: str) -> np.ndarray:
    """Mean of a clip's MERT window vectors: (n_windows, 768) -> (768,)."""
    return load_feature("mert", uid).mean(axis=0)


def load_split(split: str):
    """Return (meta, X, y_va, y_cls) for 'train', 'val' or 'test' of the mood manifest."""
    df = load_manifest("mood")
    df = df[df["split"] == split].reset_index(drop=True)
    X = np.stack([pooled_embedding(u) for u in df["uid"]]).astype(np.float32)
    y_va = df[TARGETS].to_numpy(dtype=np.float32)
    y_cls = df["mood"].map({c: i for i, c in enumerate(MOOD_CLASSES)}).to_numpy()
    return df[["uid", "dataset"]], X, y_va, y_cls