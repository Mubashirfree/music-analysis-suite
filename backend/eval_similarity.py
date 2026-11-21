"""How good are the neighbours? Evidence for objective 3 (similarity search)."""
import json

import numpy as np

from backend.similarity import VEC, _prepare
from offline.config import DOCS_DIR
from offline.features.cache import load_feature, load_manifest

K = 10


def neighbour_rows(df):
    """For every song, the row numbers of its K nearest other songs."""
    x = np.stack([load_feature(VEC, u) for u in df.uid])
    xn = _prepare(x, x.mean(axis=0))
    sims = xn @ xn.T
    np.fill_diagonal(sims, -np.inf)
    return np.argsort(-sims, axis=1)[:, :K]


def main():
    out = {"k": K}

    genre = load_manifest("genre")
    nn = neighbour_rows(genre)
    labels = genre["genre"].to_numpy()
    same = labels[nn] == labels[:, None]
    out["genre"] = {
        "songs": len(genre),
        "precision_at_10": round(float(same.mean()), 4),
        "chance": 0.10,
        "per_genre": {g: round(float(same[labels == g].mean()), 3) for g in sorted(set(labels))},
    }

    mood = load_manifest("mood")
    nn = neighbour_rows(mood)
    vals = mood[["valence_mean", "arousal_mean"]].to_numpy()
    nbr = vals[nn].mean(axis=1)
    rnd = np.random.default_rng(42).integers(0, len(mood), size=nn.shape)
    rand = vals[rnd].mean(axis=1)
    ds = mood["dataset"].to_numpy()
    out["mood"] = {
        "songs": len(mood),
        "mae_neighbours_valence_arousal": [round(float(v), 4) for v in np.abs(nbr - vals).mean(axis=0)],
        "mae_random_valence_arousal": [round(float(v), 4) for v in np.abs(rand - vals).mean(axis=0)],
        "r_neighbour_mean_valence_arousal": [round(float(np.corrcoef(nbr[:, i], vals[:, i])[0, 1]), 3) for i in (0, 1)],
        "share_neighbours_same_dataset": round(float((ds[nn] == ds[:, None]).mean()), 3),
    }

    path = DOCS_DIR / "results" / "similarity_eval.json"
    path.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print("saved", path)


if __name__ == "__main__":
    main()