"""Inference API for the backend (Phase 2). One audio file in, plain Python values out.

    predict_genre(path) -> {"genre": str, "probabilities": {genre: p, ...}}
    predict_mood(path)  -> {"valence": 0-1, "arousal": 0-1, "quadrant": str,
                            "quadrant_probabilities": {quadrant: p, ...}}
    embed(path)         -> list of 768 floats (mean MERT embedding, for similarity search)
    analyse(path)       -> {"genre": ..., "mood": ..., "embedding": ...} in one call

Genre recipe: trim edge silence, centre 30 s, peak-normalise, 3 s log-mel segments
(1.5 s hop), standardise with the checkpoint statistics, CNN with ECA, average softmax.
Mood recipe: no trim, no crop, peak-normalise, MERT 5 s windows (2.5 s hop), mean over
windows, standardise with the checkpoint statistics, multitask MLP.
Model files (git-ignored) must exist in models/: genre_cnn_eca.pt and mood_multitask.pt.
"""
import json
import sys
from functools import lru_cache

import numpy as np
import torch

from offline.config import MODELS_DIR
from offline.features.logmel import logmel_segments
from offline.features.mert import embed_song, load_mert
from offline.preprocessing.audio import preprocess
from offline.training.models import GenreCNN
from offline.training.mood_net import MultitaskMLP

GENRE_CKPT = MODELS_DIR / "genre_cnn_eca.pt"
MOOD_CKPT = MODELS_DIR / "mood_multitask.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _as_stored(y: np.ndarray) -> np.ndarray:
    """Training stored cleaned audio as float16; do the same so inputs match exactly."""
    return y.astype(np.float16).astype(np.float32)


@lru_cache(maxsize=1)
def _genre_model():
    # weights_only=False is safe here: these are our own checkpoint files.
    ckpt = torch.load(GENRE_CKPT, map_location=DEVICE, weights_only=False)
    model = GenreCNN(use_eca=ckpt["use_eca"]).to(DEVICE)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt["mean"], ckpt["std"], ckpt["genres"]


@lru_cache(maxsize=1)
def _mood_model():
    ckpt = torch.load(MOOD_CKPT, map_location=DEVICE, weights_only=False)
    hidden = ckpt["hidden"]
    model = MultitaskMLP(hidden=hidden, trunk_out=hidden // 2, dropout=ckpt["dropout"]).to(DEVICE)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt["mean"], ckpt["std"], ckpt["classes"]


@lru_cache(maxsize=1)
def _mert():
    return load_mert(DEVICE)


@torch.no_grad()
def predict_genre(path: str) -> dict:
    model, mean, std, genres = _genre_model()
    y = _as_stored(preprocess(str(path)))
    segs = logmel_segments(y).astype(np.float32)               # (n_segments, 128, 130)
    x = torch.from_numpy((segs - mean) / std).unsqueeze(1).to(DEVICE)
    probs = torch.softmax(model(x).float(), dim=1).mean(dim=0).cpu().numpy()
    return {
        "genre": genres[int(probs.argmax())],
        "probabilities": {g: round(float(p), 4) for g, p in zip(genres, probs)},
    }


def _clip_vector(path: str) -> np.ndarray:
    """Mean MERT vector (768,) of the whole clip, using the mood recipe."""
    model, extractor = _mert()
    y = _as_stored(preprocess(str(path), crop_seconds=None, trim=False))
    return embed_song(y, model, extractor, DEVICE).mean(axis=0)


@torch.no_grad()
def _mood_from_vector(vec: np.ndarray) -> dict:
    model, mean, std, classes = _mood_model()
    x = torch.from_numpy(((vec - mean) / std).astype(np.float32)).unsqueeze(0).to(DEVICE)
    va, logits = model(x)
    va = va[0].cpu().numpy()
    probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
    return {
        "valence": round(float(va[0]), 4),
        "arousal": round(float(va[1]), 4),
        "quadrant": classes[int(probs.argmax())],
        "quadrant_probabilities": {c: round(float(p), 4) for c, p in zip(classes, probs)},
    }


def predict_mood(path: str) -> dict:
    return _mood_from_vector(_clip_vector(path))


def embed(path: str) -> list:
    return [float(v) for v in _clip_vector(path)]


def analyse(path: str) -> dict:
    vec = _clip_vector(path)
    return {
        "genre": predict_genre(path),
        "mood": _mood_from_vector(vec),
        "embedding": [float(v) for v in vec],
    }


def main() -> None:
    if len(sys.argv) != 2:
        print("usage: python -m offline.inference <audio file>")
        return
    out = analyse(sys.argv[1])
    out["embedding"] = f"{len(out['embedding'])} numbers"
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()