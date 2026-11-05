import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns

from offline.config import DOCS_DIR
from offline.data.mood import MOOD_CLASSES, add_mood_category, load_mood

FIG_DIR = DOCS_DIR / "figures"


def plot_scatter(df) -> None:
    plt.figure(figsize=(7, 7))
    sns.scatterplot(data=df, x="valence_mean", y="arousal_mean", hue="dataset", alpha=0.5, s=20)
    plt.axvline(0.5, color="gray", ls="--")
    plt.axhline(0.5, color="gray", ls="--")
    for label, x, y in [("Happy/Excited", 0.62, 0.96), ("Angry/Tense", 0.02, 0.96),
                        ("Sad/Depressed", 0.02, 0.02), ("Calm/Relaxed", 0.62, 0.02)]:
        plt.text(x, y, label, weight="bold", alpha=0.7)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel("Valence (normalised)")
    plt.ylabel("Arousal (normalised)")
    plt.title("Valence-arousal space, DEAM vs PMEmo")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "03_valence_arousal_scatter.png", dpi=150)
    plt.close()


def plot_distributions(df) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, col in zip(axes, ["valence_mean", "arousal_mean"]):
        sns.histplot(data=df, x=col, hue="dataset", bins=30, kde=True, stat="density",
                     common_norm=False, ax=ax)
        ax.set_title(col)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "04_va_distributions.png", dpi=150)
    plt.close()


def plot_mood_balance(df) -> None:
    plt.figure(figsize=(8, 4))
    sns.countplot(data=df, x="mood", hue="dataset", order=MOOD_CLASSES)
    plt.title("Mood categories from valence-arousal quadrants")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "05_mood_balance.png", dpi=150)
    plt.close()


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    df = add_mood_category(load_mood())
    plot_scatter(df)
    plot_distributions(df)
    plot_mood_balance(df)

    print("Mood category counts:")
    print(df.groupby(["dataset", "mood"]).size().unstack(fill_value=0))
    print("\nShare per dataset (%):")
    print((df.groupby("dataset")["mood"].value_counts(normalize=True) * 100).round(1).unstack(fill_value=0))
    for name, g in df.groupby("dataset"):
        print(f"{name}: corr(valence, arousal) = {g['valence_mean'].corr(g['arousal_mean']):.2f}")
    print(f"Saved mood figures to {FIG_DIR}")


if __name__ == "__main__":
    main()