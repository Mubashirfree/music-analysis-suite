import librosa
import numpy as np

from offline.config import CLIP_SECONDS, SAMPLE_RATE, SEGMENT_HOP_SECONDS, SEGMENT_SECONDS

TRIM_TOP_DB = 30     # anything 30 dB below the loudest part counts as silence
MIN_PEAK = 1e-4      # below this the clip is treated as silent / broken


def load_audio(path: str) -> np.ndarray:
    """Load a file as mono, resampled to the project sample rate."""
    y, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    return y


def trim_silence(y: np.ndarray) -> np.ndarray:
    """Remove silence at the start and end only."""
    trimmed, _ = librosa.effects.trim(y, top_db=TRIM_TOP_DB)
    return trimmed


def centre_crop(y: np.ndarray, max_seconds: float = CLIP_SECONDS) -> np.ndarray:
    """Keep at most max_seconds, taken from the middle of the clip."""
    max_len = int(max_seconds * SAMPLE_RATE)
    if len(y) <= max_len:
        return y
    start = (len(y) - max_len) // 2
    return y[start:start + max_len]


def peak_normalise(y: np.ndarray) -> np.ndarray:
    """Scale so the loudest sample has magnitude 1."""
    if len(y) == 0 or np.max(np.abs(y)) < MIN_PEAK:
        raise ValueError("clip is empty or silent")
    return y / np.max(np.abs(y))


def preprocess(path: str) -> np.ndarray:
    """Full pipeline for one file: load, trim, crop, normalise."""
    y = load_audio(path)
    y = trim_silence(y)
    y = centre_crop(y)
    y = peak_normalise(y)
    return y.astype(np.float32)


def segment_starts(n_samples: int, seg_seconds: float = SEGMENT_SECONDS,
                   hop_seconds: float = SEGMENT_HOP_SECONDS) -> list[int]:
    """Start sample of every window of seg_seconds, moving by hop_seconds."""
    seg = int(seg_seconds * SAMPLE_RATE)
    hop = int(hop_seconds * SAMPLE_RATE)
    if n_samples < seg:
        return [0]
    return list(range(0, n_samples - seg + 1, hop))


def cut_segments(y: np.ndarray, seg_seconds: float = SEGMENT_SECONDS,
                 hop_seconds: float = SEGMENT_HOP_SECONDS) -> np.ndarray:
    """Cut a waveform into overlapping segments, shape (n_segments, seg_samples)."""
    seg = int(seg_seconds * SAMPLE_RATE)
    if len(y) < seg:
        y = np.pad(y, (0, seg - len(y)))
    return np.stack([y[s:s + seg] for s in segment_starts(len(y), seg_seconds, hop_seconds)])