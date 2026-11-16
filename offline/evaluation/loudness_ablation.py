import json

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from offline.config import DOCS_DIR
from offline.features.cache import load_feature
from offline.training.mood_data import load_split

ALPHAS = [1.0, 10.0, 100.0, 1000.0, 10000.0]
DIMS = ["valence", "arousal"]


def load(split: str):
    meta, X, y, _ = load_split(split)
    loud = np.array([load_feature("loudness", u)[0] for u in meta["uid"]], dtype=np.float64)
    return meta["dataset"].to_numpy(), X, y, loud


def corr(a, b) -> float:
    return round(float(np.corrcoef(a, b)[0, 1]), 3)


def main() -> None:
    parts = {s: load(s) for s in ["train", "val", "test"]}
    scaler = StandardScaler().fit(parts["train"][1])
    Z = {s: scaler.transform(parts[s][1]) for s in parts}

    best_alpha, best_err = None, 1e9
    for a in ALPHAS:
        r = Ridge(alpha=a).fit(Z["train"], parts["train"][2])
        err = np.sqrt(((parts["val"][2] - np.clip(r.predict(Z["val"]), 0, 1)) ** 2).mean(axis=0)).mean()
        if err < best_err:
            best_alpha, best_err = a, err
    ridge = Ridge(alpha=best_alpha).fit(Z["train"], parts["train"][2])

    all_ds = np.concatenate([parts[s][0] for s in parts])
    all_y = np.concatenate([parts[s][2] for s in parts])
    all_loud = np.concatenate([parts[s][3] for s in parts])

    held = ["val", "test"]  # songs the Ridge did not train on
    h_ds = np.concatenate([parts[s][0] for s in held])
    h_y = np.concatenate([parts[s][2] for s in held])
    h_loud = np.concatenate([parts[s][3] for s in held])
    h_pred = np.clip(ridge.predict(np.vstack([Z[s] for s in held])), 0, 1)
    resid = h_y - h_pred

    out = {}
    print("dataset | mean loudness dB | loudness vs label r (V, A) | loudness vs MERT-Ridge error r (V, A)")
    for name in ["all", "deam", "pmemo"]:
        m = np.ones(len(all_ds), bool) if name == "all" else all_ds == name
        mh = np.ones(len(h_ds), bool) if name == "all" else h_ds == name
        row = {
            "mean_loudness_db": round(float(all_loud[m].mean()), 2),
            "loudness_vs_label_r": [corr(all_loud[m], all_y[m, i]) for i in range(2)],
            "loudness_vs_residual_r": [corr(h_loud[mh], resid[mh, i]) for i in range(2)],
        }
        out[name] = row
        print(f"{name:5s} | {row['mean_loudness_db']:7.2f} | {row['loudness_vs_label_r']} | "
              f"{row['loudness_vs_residual_r']}")

    with open(DOCS_DIR / "results" / "loudness_ablation.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()