import argparse
import copy
import json
import random

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import ConfusionMatrixDisplay
from torch.utils.data import DataLoader, TensorDataset

from offline.config import DOCS_DIR, MODELS_DIR, SEED
from offline.data.mood import MOOD_CLASSES
from offline.evaluation.mood_metrics import classification_metrics, full_report, quadrant_from_va
from offline.training.mood_data import load_split
from offline.training.mood_net import MultitaskMLP

FIG_DIR = DOCS_DIR / "figures"
RESULTS_DIR = DOCS_DIR / "results"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tasks", choices=["both", "reg", "cls"], default="both")
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=1e-1)
    p.add_argument("--patience", type=int, default=30)
    p.add_argument("--reg-weight", type=float, default=10.0)
    p.add_argument("--cls-weight", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--suffix", type=str, default="")
    return p.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def predict(model, X: np.ndarray, device: str):
    model.eval()
    va, logits = model(torch.from_numpy(X).to(device))
    return va.cpu().numpy(), logits.cpu().numpy()


def evaluate(model, X, y, c, meta, tasks: str, device: str):
    """Return (report, predicted valence/arousal, predicted quadrant)."""
    va, logits = predict(model, X, device)
    cls_pred = quadrant_from_va(va) if tasks == "reg" else logits.argmax(axis=1)
    y_pred = None if tasks == "cls" else va
    return full_report(meta, y, y_pred, c, cls_pred), va, cls_pred


def val_score(report: dict, tasks: str) -> float:
    """Higher is better: macro-F1 for the category-only model, else negative mean RMSE."""
    if tasks == "cls":
        return report["all"]["classification"]["macro_f1"]
    reg = report["all"]["regression"]
    return -(reg["valence"]["rmse"] + reg["arousal"]["rmse"]) / 2


def show(tag: str, split: str, report: dict) -> None:
    for d in ["all", "deam", "pmemo"]:
        r = report[d]
        parts = [f"{tag:12s} {split:4s} {d:5s} n={r['n']:4d}"]
        if "regression" in r:
            reg = r["regression"]
            parts.append(f"V rmse {reg['valence']['rmse']:.3f} r {reg['valence']['pearson_r']:.3f}")
            parts.append(f"A rmse {reg['arousal']['rmse']:.3f} r {reg['arousal']['pearson_r']:.3f}")
        if "classification" in r:
            parts.append(f"macro-F1 {r['classification']['macro_f1']:.3f}")
        print(" | ".join(parts))


def plot_scatter(meta, y_true, y_pred, tag: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for i, name in enumerate(["valence", "arousal"]):
        for ds in sorted(meta["dataset"].unique()):
            m = (meta["dataset"] == ds).to_numpy()
            axes[i].scatter(y_true[m, i], y_pred[m, i], s=12, alpha=0.6, label=ds)
        axes[i].plot([0, 1], [0, 1], "k--", lw=1)
        axes[i].set_xlabel(f"true {name}")
        axes[i].set_ylabel(f"predicted {name}")
        axes[i].set_title(f"{name}, test set")
        axes[i].legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"12_mood_scatter_{tag}.png", dpi=150)
    plt.close()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    name = "multitask" if args.tasks == "both" else args.tasks
    tag = name if args.seed == SEED else f"{name}_s{args.seed}"
    tag=tag + args.suffix

    meta, X, y, c = {}, {}, {}, {}
    for s in ["train", "val", "test"]:
        meta[s], X[s], y[s], c[s] = load_split(s)
        c[s] = c[s].astype(np.int64)
    mean = X["train"].mean(axis=0)
    std = X["train"].std(axis=0) + 1e-6
    X = {s: ((X[s] - mean) / std).astype(np.float32) for s in X}
    print(f"device {device} | model {tag} | train {len(X['train'])} val {len(X['val'])} "
          f"test {len(X['test'])}")

    counts = np.bincount(c["train"], minlength=len(MOOD_CLASSES))
    class_w = torch.tensor(len(c["train"]) / (len(MOOD_CLASSES) * counts),
                           dtype=torch.float32, device=device)

    train_ds = TensorDataset(torch.from_numpy(X["train"]), torch.from_numpy(y["train"]),
                             torch.from_numpy(c["train"]))
    loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True)

    model = MultitaskMLP(hidden=args.hidden, trunk_out=args.hidden // 2,
                         dropout=args.dropout).to(device)    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

    best_score, best_state, bad_epochs = -1e9, None, 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for xb, yb, cb in loader:
            xb, yb, cb = xb.to(device), yb.to(device), cb.to(device)
            optimizer.zero_grad(set_to_none=True)
            va, logits = model(xb)
            loss = torch.zeros((), device=device)
            if args.tasks in ("both", "reg"):
                loss = loss + args.reg_weight * F.smooth_l1_loss(va, yb, beta=0.1)
            if args.tasks in ("both", "cls"):
                loss = loss + args.cls_weight * F.cross_entropy(logits, cb, weight=class_w)
            loss.backward()
            optimizer.step()
            running += loss.item() * len(yb)

        val_report, _, _ = evaluate(model, X["val"], y["val"], c["val"], meta["val"],
                                    args.tasks, device)
        score = val_score(val_report, args.tasks)
        if epoch == 1 or epoch % 10 == 0:
            print(f"epoch {epoch:3d} | loss {running / len(train_ds):.4f} | val score {score:.4f}")
        if score > best_score:
            best_score, best_state, bad_epochs = score, copy.deepcopy(model.state_dict()), 0
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"early stop at epoch {epoch}: no improvement for {args.patience} epochs")
                break

    print(f"best val score {best_score:.4f}")
    model.load_state_dict(best_state)
    val_report, _, _ = evaluate(model, X["val"], y["val"], c["val"], meta["val"], args.tasks, device)
    test_report, test_va, test_cls = evaluate(model, X["test"], y["test"], c["test"],
                                              meta["test"], args.tasks, device)
    show(tag, "val", val_report)
    show(tag, "test", test_report)

    results = {"args": vars(args), "best_val_score": round(best_score, 4),
               "val": val_report, "test": test_report}
    if args.tasks == "both":
        results["test_quadrant_from_regression"] = classification_metrics(
            c["test"], quadrant_from_va(test_va))
        print("quadrant macro-F1 from the regression head cut at 0.5:",
              results["test_quadrant_from_regression"]["macro_f1"])
    with open(RESULTS_DIR / f"mood_{tag}.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    torch.save({"state_dict": model.state_dict(), "mean": mean, "std": std,
                "classes": MOOD_CLASSES, "tasks": args.tasks,
                "hidden": args.hidden, "dropout": args.dropout},
               MODELS_DIR / f"mood_{tag}.pt")
    if args.tasks != "reg":
        fig, ax = plt.subplots(figsize=(7, 6))
        ConfusionMatrixDisplay.from_predictions(
            c["test"], test_cls, labels=list(range(len(MOOD_CLASSES))),
            display_labels=MOOD_CLASSES, xticks_rotation=30, cmap="Blues",
            colorbar=False, ax=ax)
        ax.set_title(f"Mood quadrants ({tag}), test set")
        plt.tight_layout()
        plt.savefig(FIG_DIR / f"11_mood_confusion_{tag}.png", dpi=150)
        plt.close()
    if args.tasks != "cls":
        plot_scatter(meta["test"], y["test"], test_va, tag)


if __name__ == "__main__":
    main()