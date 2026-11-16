import numpy as np

from offline.config import SAMPLE_RATE
from offline.preprocessing.audio import centre_crop, cut_segments


def test_centre_crop_none_keeps_whole_clip():
    y = np.zeros(45 * SAMPLE_RATE, dtype=np.float32)
    assert len(centre_crop(y, None)) == len(y)


def test_45s_clip_gives_17_mert_windows():
    y = np.zeros(45 * SAMPLE_RATE, dtype=np.float32)
    assert cut_segments(y, 5.0, 2.5).shape[0] == 17