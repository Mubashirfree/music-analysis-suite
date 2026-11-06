import numpy as np

from offline.features.handcrafted import handcrafted_summary
from offline.config import N_MELS, SAMPLE_RATE
from offline.features.chroma import chroma_summary
from offline.features.logmel import logmel_segments


def _tone(seconds=30, freq=440.0):
    t = np.arange(seconds * SAMPLE_RATE) / SAMPLE_RATE
    return np.sin(2 * np.pi * freq * t).astype(np.float32)


def test_chroma_has_24_values():
    assert chroma_summary(_tone()).shape == (24,)


def test_chroma_peaks_at_note_a_for_440hz():
    mean_part = chroma_summary(_tone())[:12]
    assert int(np.argmax(mean_part)) == 9   # index 9 is the note A


def test_logmel_shape():
    out = logmel_segments(_tone())
    assert out.shape[0] == 19 and out.shape[1] == N_MELS

def test_handcrafted_has_49_values():
    out = handcrafted_summary(_tone(seconds=10))
    assert out.shape == (49,)
    assert np.isfinite(out).all()