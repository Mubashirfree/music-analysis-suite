import argparse
import json
import random
import time

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import ConfusionMatrixDisplay, f1_score
from torch import nn
from torch.utils.data import DataLoader

from offline.config import DOCS_DIR, GENRES, MODELS_DIR, SEED
from offline.training.genre_data import GenreSegments
from offline.training.models import GenreCNN

FIG_DIR = DOCS_DIR / "figures"
RESULTS_DIR = DOCS_DIR / "results"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--no-eca", action="store_true")
    p.add_argument("--seed", type=int, default=SEED)
    return p.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def predict(model, loader, device: str, n_songs: int):
    """Return segment accuracy and per-song class probabilities (mean over its segments)."""
    model.eval()
    prob_sum = torch.zeros(n_songs, len(GENRES))
    counts = torch.zeros(n_songs)
    correct, total = 0, 0
    for x, y, song in loader:
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(device == "cuda")):
            logits = model(x.to(device))
        p = torch.softmax(logits.float(), dim=1).cpu()
        correct += int((p.argmax(dim=1) == y).sum())
        total += len(y)
        prob_sum.index_add_(0, song, p)
        counts.index_add_(0, song, torch.ones(len(song)))
    return correct / total, prob_sum / counts.unsqueeze(1)


def song_metrics(song_probs, labels):
    pred = song_probs.argmax(dim=1).numpy()
    return {
        "song_accuracy": round(float((pred == labels).mean()), 4),
        "song_macro_f1": round(float(f1_score(labels, pred, average="macro")), 4),
    }, pred


def plot_history(history: dict, tag: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(history["train_loss"])
    axes[0].set_title("Training loss")
    axes[0].set_xlabel("epoch")
    axes[1].plot(history["val_seg_acc"], label="val segment accuracy")
    axes[1].plot(history["val_song_acc"], label="val song accuracy (voting)")
    axes[1].set_title("Validation accuracy")
    axes[1].set_xlabel("epoch")
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"09_cnn_training_{tag}.png", dpi=150)
    plt.close()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = device == "cuda"
    tag = "noeca" if args.no_eca else "eca"
    run = tag if args.seed == SEED else f"{tag}_s{args.seed}"

    train_ds = GenreSegments("train", augment=True)
    val_ds = GenreSegments("val", train_ds.mean, train_ds.std)
    test_ds = GenreSegments("test", train_ds.mean, train_ds.std)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, pin_memory=use_amp)
    val_loader = DataLoader(val_ds, batch_size=128, pin_memory=use_amp)
    test_loader = DataLoader(test_ds, batch_size=128, pin_memory=use_amp)
    print(f"device {device} | model {run} | segments train {len(train_ds)} "
          f"val {len(val_ds)} test {len(test_ds)}")

    model = GenreCNN(use_eca=not args.no_eca).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"parameters: {n_params:,}")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, total_steps=args.epochs * len(train_loader))
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    ckpt_path = MODELS_DIR / f"genre_cnn_{run}.pt"
    history = {"train_loss": [], "val_seg_acc": [], "val_song_acc": []}
    best, bad_epochs = -1.0, 0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        running = 0.0
        for x, y, _ in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                loss = criterion(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            running += loss.item() * len(y)
        train_loss = running / len(train_ds)

        seg_acc, val_probs = predict(model, val_loader, device, len(val_ds.songs))
        val_m, _ = song_metrics(val_probs, val_ds.labels)
        history["train_loss"].append(train_loss)
        history["val_seg_acc"].append(seg_acc)
        history["val_song_acc"].append(val_m["song_accuracy"])
        print(f"epoch {epoch:2d} | loss {train_loss:.3f} | val seg acc {seg_acc:.3f} "
              f"| val song acc {val_m['song_accuracy']:.3f} | {time.time() - t0:.0f}s")

        if val_m["song_accuracy"] > best:
            best, bad_epochs = val_m["song_accuracy"], 0
            torch.save({"state_dict": model.state_dict(), "mean": train_ds.mean,
                        "std": train_ds.std, "genres": GENRES, "use_eca": not args.no_eca},
                       ckpt_path)
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"early stop: no improvement for {args.patience} epochs")
                break

    plot_history(history, run)

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    seg_acc, test_probs = predict(model, test_loader, device, len(test_ds.songs))
    test_m, test_pred = song_metrics(test_probs, test_ds.labels)
    test_m["segment_accuracy"] = round(seg_acc, 4)
    test_m["best_val_song_accuracy"] = best
    test_m["parameters"] = n_params
    test_m["seed"] = args.seed

    fig, ax = plt.subplots(figsize=(8, 7))
    ConfusionMatrixDisplay.from_predictions(
        test_ds.labels, test_pred, labels=list(range(len(GENRES))), display_labels=GENRES,
        xticks_rotation=45, cmap="Blues", colorbar=False, ax=ax)
    ax.set_title(f"CNN ({run}), test set, song-level voting")
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"10_cnn_confusion_{run}.png", dpi=150)
    plt.close()

    with open(RESULTS_DIR / f"cnn_genre_{run}.json", "w", encoding="utf-8") as f:
        json.dump(test_m, f, indent=2)

    print(f"\nTEST ({run}): song acc {test_m['song_accuracy']:.3f} | "
          f"macro-F1 {test_m['song_macro_f1']:.3f} | segment acc {seg_acc:.3f}")
    rf_file = RESULTS_DIR / "rf_baseline.json"
    if rf_file.exists():
        rf = json.load(open(rf_file, encoding="utf-8"))["chroma_plus_handcrafted"]["test"]
        print(f"Random Forest baseline: acc {rf['accuracy']:.3f} | macro-F1 {rf['macro_f1']:.3f}")


if __name__ == "__main__":
    main()