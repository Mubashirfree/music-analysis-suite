import numpy as np
import torch
from torch.utils.data import Dataset

from offline.features.cache import load_feature, load_manifest


def compute_stats(songs: list) -> tuple:
    """Mean and standard deviation over every value of every song (float64 sums)."""
    n, s, ss = 0, 0.0, 0.0
    for a in songs:
        a = a.astype(np.float64)
        n += a.size
        s += a.sum()
        ss += (a ** 2).sum()
    mean = s / n
    return float(mean), float(np.sqrt(ss / n - mean ** 2))


def spec_augment(x: np.ndarray, max_freq: int = 16, max_time: int = 20) -> np.ndarray:
    """Blank one random band of mel bins and one random stretch of time frames."""
    x = x.copy()
    f = np.random.randint(0, max_freq + 1)
    f0 = np.random.randint(0, x.shape[0] - f + 1)
    x[f0:f0 + f, :] = 0.0
    t = np.random.randint(0, max_time + 1)
    t0 = np.random.randint(0, x.shape[1] - t + 1)
    x[:, t0:t0 + t] = 0.0
    return x


class GenreSegments(Dataset):
    """Every 3 s log-mel segment of every song in one split."""

    def __init__(self, split: str, mean: float = None, std: float = None, augment: bool = False):
        df = load_manifest("genre")
        df = df[df["split"] == split].reset_index(drop=True)
        self.labels = df["label"].to_numpy()
        self.songs = [load_feature("logmel", uid) for uid in df["uid"]]   # float16 arrays
        self.index = [(i, j) for i, s in enumerate(self.songs) for j in range(len(s))]
        if mean is None:
            mean, std = compute_stats(self.songs)
        self.mean, self.std, self.augment = mean, std, augment

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, k: int):
        i, j = self.index[k]
        x = (self.songs[i][j].astype(np.float32) - self.mean) / self.std
        if self.augment:
            x = spec_augment(x)
        return torch.from_numpy(x).unsqueeze(0), int(self.labels[i]), i