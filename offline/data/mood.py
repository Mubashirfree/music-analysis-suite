import pandas as pd

from offline.data.deam import load_deam
from offline.data.pmemo import load_pmemo

# Source scale (min, max) of each dataset's valence/arousal annotations.
SCALES = {
    "deam": (1.0, 9.0),
    "pmemo": (0.0, 1.0),
}


def normalise(df: pd.DataFrame, dataset: str) -> pd.DataFrame:
    """Rescale valence and arousal of one dataset to the common range 0-1."""
    lo, hi = SCALES[dataset]
    out = df.copy()
    for col in ("valence_mean", "arousal_mean"):
        out[col] = (out[col] - lo) / (hi - lo)
    return out


def load_mood() -> pd.DataFrame:
    """Load every mood dataset, normalise to 0-1, and stack into one table."""
    frames = [
        normalise(load_deam(), "deam"),
        normalise(load_pmemo(), "pmemo"),
    ]
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    df = load_mood()
    print(f"Combined mood table: {len(df)} songs")
    print(df.groupby("dataset")[["valence_mean", "arousal_mean"]].agg(["mean", "std", "min", "max"]).round(3))


if __name__ == "__main__":
    main()