import numpy as np

from offline.config import SAMPLE_RATE
from offline.preprocessing.audio import centre_crop, cut_segments, peak_normalise


def test_peak_normalise_gives_peak_one():
    y = 0.2 * np.sin(np.linspace(0, 100, SAMPLE_RATE))
    assert abs(np.max(np.abs(peak_normalise(y))) - 1.0) < 1e-6


def test_30s_clip_gives_19_segments():
    y = np.random.randn(30 * SAMPLE_RATE).astype(np.float32)
    segs = cut_segments(y)
    assert segs.shape == (19, 3 * SAMPLE_RATE)


def test_short_clip_is_padded_to_one_segment():
    y = np.ones(SAMPLE_RATE, dtype=np.float32)
    assert cut_segments(y).shape == (1, 3 * SAMPLE_RATE)


def test_centre_crop_limits_length():
    y = np.zeros(45 * SAMPLE_RATE, dtype=np.float32)
    assert len(centre_crop(y)) == 30 * SAMPLE_RATE