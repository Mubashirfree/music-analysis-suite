from pathlib import Path

import pandas as pd

from offline.config import DATA_RAW

DEAM_DIR = DATA_RAW / "deam"
AUDIO_DIR = DEAM_DIR / "DEAM_audio" / "MEMD_audio"
ANNOTATIONS_CSV = (
    DEAM_DIR
    / "DEAM_Annotations"
    / "annotations"
    / "annotations averaged per song"
    / "song_level"
    / "static_annotations_averaged_songs_1_2000.csv"
)


def load_deam(csv_path: Path = ANNOTATIONS_CSV, audio_dir: Path = AUDIO_DIR) -> pd.DataFrame:
    """Return one row per DEAM song: audio path plus mean valence and arousal (1-9 scale)."""
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    df = df[["song_id", "valence_mean", "arousal_mean"]].dropna()
    df["song_id"] = df["song_id"].astype(int)

    df["path"] = df["song_id"].map(lambda i: str(audio_dir / f"{i}.mp3"))
    has_audio = df["path"].map(lambda p: Path(p).exists())
    n_missing = int((~has_audio).sum())
    if n_missing:
        print(f"Note: {n_missing} annotated songs have no audio file and were dropped")
    df = df[has_audio].copy()

    df["dataset"] = "deam"
    return df.reset_index(drop=True)


def main() -> None:
    df = load_deam()
    print(f"Loaded {len(df)} DEAM songs")
    print(df[["valence_mean", "arousal_mean"]].describe().round(2))
    print(df.head())


if __name__ == "__main__":
    main()