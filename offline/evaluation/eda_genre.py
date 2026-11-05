import matplotlib

matplotlib.use("Agg")  # draw to files, no display needed in WSL

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from offline.config import DOCS_DIR, GENRES, SAMPLE_RATE
from offline.data.gtzan import load_gtzan

FIG_DIR = DOCS_DIR / "figures"


def plot_class_balance(df) -> None:
    plt.figure(figsize=(9, 4))
    sns.countplot(data=df, x="genre", order=GENRES, color="#4C72B0")
    plt.title("GTZAN: clips per genre")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "01_genre_balance.png", dpi=150)
    plt.close()


def plot_mel_per_genre(df) -> None:
    fig, axes = plt.subplots(2, 5, figsize=(20, 7))
    for ax, genre in zip(axes.ravel(), GENRES):
        path = df[df["genre"] == genre].iloc[0]["path"]
        y, sr = librosa.load(path, sr=SAMPLE_RATE, duration=30)
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        librosa.display.specshow(mel_db, sr=sr, x_axis="time", y_axis="mel", ax=ax)
        ax.set_title(genre)
    plt.suptitle("Mel spectrogram, one example per genre")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "02_mel_per_genre.png", dpi=120)
    plt.close()


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    df = load_gtzan()
    plot_class_balance(df)
    plot_mel_per_genre(df)
    print(f"Saved genre figures to {FIG_DIR}")


if __name__ == "__main__":
    main()