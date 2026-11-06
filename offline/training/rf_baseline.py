import json

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, f1_score

from offline.config import DOCS_DIR, GENRES, SEED
from offline.features.cache import load_feature, load_manifest

FIG_DIR = DOCS_DIR / "figures"
RESULTS_DIR = DOCS_DIR / "results"
LABELS = list(range(len(GENRES)))

SETUPS = {
    "chroma_only": ["chroma"],
    "chroma_plus_handcrafted": ["chroma", "handcrafted"],
}


def build_matrix(df, feature_names):
    """Stack the cached features of every song in df into a 2D array (songs x features)."""
    rows = [
        np.concatenate([load_feature(name, uid) for name in feature_names])
        for uid in df["uid"]
    ]
    return np.stack(rows)


def run_setup(df, feature_names):
    """Train on the train split, evaluate on val and test. Returns model, metrics, test predictions."""
    parts = {s: df[df["split"] == s] for s in ("train", "val", "test")}
    X = {s: build_matrix(p, feature_names) for s, p in parts.items()}
    y = {s: p["label"].to_numpy() for s, p in parts.items()}

    model = RandomForestClassifier(n_estimators=500, random_state=SEED, n_jobs=-1)
    model.fit(X["train"], y["train"])

    metrics = {"n_features": int(X["train"].shape[1])}
    preds = {}
    for s in ("val", "test"):
        preds[s] = model.predict(X[s])
        metrics[s] = {
            "accuracy": round(float(accuracy_score(y[s], preds[s])), 4),
            "macro_f1": round(float(f1_score(y[s], preds[s], average="macro")), 4),
        }
    return model, metrics, y["test"], preds["test"]


def plot_confusion(y_true, y_pred, title, filename):
    fig, ax = plt.subplots(figsize=(8, 7))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, labels=LABELS, display_labels=GENRES,
        xticks_rotation=45, cmap="Blues", colorbar=False, ax=ax,
    )
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(FIG_DIR / filename, dpi=150)
    plt.close()


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df = load_manifest("genre")
    print(f"Songs: {len(df)} | " + " | ".join(f"{s}: {(df['split'] == s).sum()}" for s in ("train", "val", "test")))

    results = {}
    for name, feature_names in SETUPS.items():
        model, metrics, y_test, p_test = run_setup(df, feature_names)
        results[name] = metrics
        print(f"\n{name} ({metrics['n_features']} features)")
        print(f"  val : acc {metrics['val']['accuracy']:.3f}  macro-F1 {metrics['val']['macro_f1']:.3f}")
        print(f"  test: acc {metrics['test']['accuracy']:.3f}  macro-F1 {metrics['test']['macro_f1']:.3f}")
        plot_confusion(y_test, p_test, f"Random Forest ({name}), test set", f"08_rf_confusion_{name}.png")

    with open(RESULTS_DIR / "rf_baseline.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {RESULTS_DIR / 'rf_baseline.json'} and figures to {FIG_DIR}")


if __name__ == "__main__":
    main()