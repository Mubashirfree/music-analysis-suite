import json

import numpy as np

from offline.config import DOCS_DIR

RESULTS = DOCS_DIR / "results"
MODES = {
    "multitask": "both heads (multitask)",
    "reg": "regression head only",
    "cls": "category head only",
}
SEEDS = ["", "_s1", "_s2"]


def ms(values) -> str:
    return f"{np.mean(values):.3f} ± {np.std(values):.3f}"


def main() -> None:
    summary = {}
    for mode, label in MODES.items():
        runs = [json.load(open(RESULTS / f"mood_{mode}{s}.json", encoding="utf-8")) for s in SEEDS]
        summary[mode] = {}
        for ds in ["all", "deam", "pmemo"]:
            tests = [r["test"][ds] for r in runs]
            row = {}
            if "regression" in tests[0]:
                for dim in ["valence", "arousal"]:
                    for metric in ["rmse", "pearson_r"]:
                        row[f"{dim}_{metric}"] = [t["regression"][dim][metric] for t in tests]
            if "classification" in tests[0]:
                row["macro_f1"] = [t["classification"]["macro_f1"] for t in tests]
            summary[mode][ds] = row
            parts = [f"{label:24s} {ds:5s}"] + [f"{k} {ms(v)}" for k, v in row.items()]
            print(" | ".join(parts))
    with open(RESULTS / "mood_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()