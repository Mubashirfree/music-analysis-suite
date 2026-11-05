import matplotlib

matplotlib.use("Agg")

import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from offline.config import DATA_PROCESSED, DOCS_DIR, SAMPLE_RATE, SEGMENT_SECONDS
from offline.preprocessing.audio import load_audio, segment_starts

FIG_DIR = DOCS_DIR / "figures"


def plot_before_after(genre: pd.DataFrame) -> None:
    rec = genre[genre["genre"] == "rock"].iloc[0]
    raw = load_audio(rec["path"])
    clean = np.load(rec["proc_path"]).astype(np.float32)

    fig, axes = plt.subplots(2, 1, figsize=(12, 5), sharex=True)
    librosa.display.waveshow(raw, sr=SAMPLE_RATE, ax=axes[0], color="#4C72B0")
    axes[0].set_title(f"Before: {rec['filename']} (loaded at {SAMPLE_RATE} Hz, mono)")
    librosa.display.waveshow(clean, sr=SAMPLE_RATE, ax=axes[1], color="#DD8452")
    axes[1].set_title("After: trimmed, cropped, peak-normalised, with 3 s segment boundaries")
    for s in segment_starts(len(clean))[::2]:
        axes[1].axvline(s / SAMPLE_RATE, color="gray", lw=0.6, ls="--")
    for ax in axes:
        ax.set_ylim(-1.05, 1.05)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "06_preprocessing_example.png", dpi=150)
    plt.close()


def plot_splits(genre: pd.DataFrame, mood: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    pd.crosstab(genre["genre"], genre["split"])[["train", "val", "test"]].plot(
        kind="bar", stacked=True, ax=axes[0])
    axes[0].set_title("Genre: songs per split")
    mood_ct = pd.crosstab(mood["dataset"] + " / " + mood["mood"], mood["split"])
    mood_ct[["train", "val", "test"]].plot(kind="bar", stacked=True, ax=axes[1])
    axes[1].set_title("Mood: songs per split (dataset / class)")
    for ax in axes:
        ax.tick_params(axis="x", rotation=60)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "07_split_sizes.png", dpi=150)
    plt.close()


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    genre = pd.read_csv(DATA_PROCESSED / "genre_manifest.csv")
    mood = pd.read_csv(DATA_PROCESSED / "mood_manifest.csv")
    plot_before_after(genre)
    plot_splits(genre, mood)
    print(f"Segment length: {SEGMENT_SECONDS}s | saved figures to {FIG_DIR}")


if __name__ == "__main__":
    main()