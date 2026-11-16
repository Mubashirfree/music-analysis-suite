import numpy as np
from sklearn.metrics import f1_score, mean_squared_error, r2_score

from offline.data.mood import MOOD_CLASSES

DIMS = ["valence", "arousal"]


def quadrant_from_va(va: np.ndarray, mid: float = 0.5) -> np.ndarray:
    """Quadrant index (0-3, same order as MOOD_CLASSES) from valence/arousal on a 0-1 scale."""
    v = va[:, 0] >= mid
    a = va[:, 1] >= mid
    return np.select([v & a, ~v & a, ~v & ~a, v & ~a], [0, 1, 2, 3])


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    out = {}
    for i, name in enumerate(DIMS):
        t, p = y_true[:, i], y_pred[:, i]
        r = 0.0 if np.std(p) < 1e-6 else float(np.corrcoef(t, p)[0, 1])
        out[name] = {
            "rmse": round(float(np.sqrt(mean_squared_error(t, p))), 4),
            "pearson_r": round(r, 4),
            "r2": round(float(r2_score(t, p)), 4),
        }
    return out


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    labels = list(range(len(MOOD_CLASSES)))
    return {
        "accuracy": round(float((y_true == y_pred).mean()), 4),
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro",
                                         labels=labels, zero_division=0)), 4),
    }


def full_report(meta, y_true, y_pred, cls_true, cls_pred) -> dict:
    """Metrics for all songs together and for each dataset separately."""
    report = {}
    for name in ["all"] + sorted(meta["dataset"].unique()):
        if name == "all":
            mask = np.ones(len(meta), dtype=bool)
        else:
            mask = (meta["dataset"] == name).to_numpy()
        row = {"n": int(mask.sum())}
        if y_pred is not None:
            row["regression"] = regression_metrics(y_true[mask], y_pred[mask])
        if cls_pred is not None:
            row["classification"] = classification_metrics(cls_true[mask], cls_pred[mask])
        report[name] = row
    return report