import json

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from offline.config import DOCS_DIR, SEED
from offline.evaluation.mood_metrics import full_report, quadrant_from_va
from offline.training.mood_data import load_split

ALPHAS = [1.0, 10.0, 100.0, 1000.0, 10000.0]


def mean_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(((y_true - y_pred) ** 2).mean(axis=0)).mean())


def show(name: str, split: str, report: dict) -> None:
    for d in ["all", "deam", "pmemo"]:
        r = report[d]
        reg, cls = r["regression"], r["classification"]
        print(f"{name:13s} {split:4s} {d:5s} n={r['n']:4d} | "
              f"V rmse {reg['valence']['rmse']:.3f} r {reg['valence']['pearson_r']:.3f} | "
              f"A rmse {reg['arousal']['rmse']:.3f} r {reg['arousal']['pearson_r']:.3f} | "
              f"macro-F1 {cls['macro_f1']:.3f}")


def main() -> None:
    meta, X, y, c = {}, {}, {}, {}
    for s in ["train", "val", "test"]:
        meta[s], X[s], y[s], c[s] = load_split(s)
    print({s: len(X[s]) for s in X})

    scaler = StandardScaler().fit(X["train"])
    Z = {s: scaler.transform(X[s]) for s in X}

    best_alpha, best_score = None, 1e9
    for a in ALPHAS:
        m = Ridge(alpha=a).fit(Z["train"], y["train"])
        score = mean_rmse(y["val"], np.clip(m.predict(Z["val"]), 0, 1))
        print(f"ridge alpha {a:g}: val mean RMSE {score:.4f}")
        if score < best_score:
            best_alpha, best_score = a, score

    models = {
        "ridge": Ridge(alpha=best_alpha).fit(Z["train"], y["train"]),
        "random_forest": RandomForestRegressor(
            n_estimators=300, n_jobs=-1, random_state=SEED).fit(Z["train"], y["train"]),
    }
    train_mean = y["train"].mean(axis=0)

    results = {"ridge_alpha": best_alpha}
    for name in ["train_mean", "ridge", "random_forest"]:
        results[name] = {}
        for s in ["val", "test"]:
            if name == "train_mean":
                pred = np.tile(train_mean, (len(y[s]), 1))
            else:
                pred = np.clip(models[name].predict(Z[s]), 0, 1)
            report = full_report(meta[s], y[s], pred, c[s], quadrant_from_va(pred))
            results[name][s] = report
            show(name, s, report)

    out = DOCS_DIR / "results" / "mood_baselines.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("saved", out)


if __name__ == "__main__":
    main()