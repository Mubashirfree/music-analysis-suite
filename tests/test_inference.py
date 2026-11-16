import numpy as np
import pytest

from offline.config import DATA_RAW
from offline.features.cache import load_feature
from offline.inference import GENRE_CKPT, MOOD_CKPT, analyse, embed

JAZZ = DATA_RAW / "gtzan" / "Data" / "genres_original" / "jazz" / "jazz.00001.wav"
DEAM_CLIP = DATA_RAW / "deam" / "DEAM_audio" / "MEMD_audio" / "2.mp3"

pytestmark = pytest.mark.skipif(
    not (GENRE_CKPT.exists() and MOOD_CKPT.exists() and JAZZ.exists() and DEAM_CLIP.exists()),
    reason="model files or audio not present",
)


def test_analyse_follows_the_contract():
    out = analyse(str(JAZZ))
    assert len(out["genre"]["probabilities"]) == 10
    assert abs(sum(out["genre"]["probabilities"].values()) - 1.0) < 0.01
    assert out["genre"]["genre"] in out["genre"]["probabilities"]
    assert 0.0 <= out["mood"]["valence"] <= 1.0 and 0.0 <= out["mood"]["arousal"] <= 1.0
    assert abs(sum(out["mood"]["quadrant_probabilities"].values()) - 1.0) < 0.01
    assert len(out["embedding"]) == 768


def test_embed_matches_the_vector_cached_during_training():
    cached = load_feature("mert", "deam_2").mean(axis=0)
    fresh = np.array(embed(str(DEAM_CLIP)))
    cos = float(cached @ fresh / (np.linalg.norm(cached) * np.linalg.norm(fresh)))
    assert cos > 0.999