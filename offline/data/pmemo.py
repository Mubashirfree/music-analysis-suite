from pathlib import Path

import pandas as pd

from offline.config import DATA_RAW

PMEMO_DIR = DATA_RAW / "pmemo"
AUDIO_DIR = PMEMO_DIR / "chorus (2)" / "chorus"
ANNOTATIONS_CSV = PMEMO_DIR / "static_annotations.csv"


def load_pmemo(csv_path: Path = ANNOTATIONS_CSV, audio_dir: Path = AUDIO_DIR) -> pd.DataFrame:
    """Return one row per PMEmo song: audio path plus mean valence and arousal (0-1 scale)."""
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    df = df.rename(columns={
        "musicId": "song_id",
        "Valence(mean)": "valence_mean",
        "Arousal(mean)": "arousal_mean",
    })
    df = df[["song_id", "valence_mean", "arousal_mean"]].dropna()
    df["song_id"] = df["song_id"].astype(int)

    df["path"] = df["song_id"].map(lambda i: str(audio_dir / f"{i}.mp3"))
    has_audio = df["path"].map(lambda p: Path(p).exists())
    n_missing = int((~has_audio).sum())
    if n_missing:
        print(f"Note: {n_missing} annotated songs have no audio file and were dropped")
    df = df[has_audio].copy()

    df["dataset"] = "pmemo"
    return df.reset_index(drop=True)


def main() -> None:
    df = load_pmemo()
    print(f"Loaded {len(df)} PMEmo songs")
    print(df[["valence_mean", "arousal_mean"]].describe().round(3))
    print(df.head())


if __name__ == "__main__":
    main()