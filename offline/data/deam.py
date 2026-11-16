from pathlib import Path

import pandas as pd

from offline.config import DATA_RAW

DEAM_DIR = DATA_RAW / "deam"
AUDIO_DIR = DEAM_DIR / "DEAM_audio" / "MEMD_audio"
SONG_LEVEL_DIR = (
    DEAM_DIR
    / "DEAM_Annotations"
    / "annotations"
    / "annotations averaged per song"
    / "song_level"
)
# Songs 1-2000 are 45 s excerpts; songs 2001-2058 are the 58 full-length songs.
ANNOTATION_FILES = [
    SONG_LEVEL_DIR / "static_annotations_averaged_songs_1_2000.csv",
    SONG_LEVEL_DIR / "static_annotations_averaged_songs_2000_2058.csv",
]
ANNOTATIONS_CSV = ANNOTATION_FILES[0]  # kept so older imports still work


def load_deam(csv_paths=ANNOTATION_FILES, audio_dir: Path = AUDIO_DIR) -> pd.DataFrame:
    """Return one row per DEAM song: audio path plus mean valence and arousal (1-9 scale)."""
    frames = []
    for csv_path in csv_paths:
        part = pd.read_csv(csv_path)
        part.columns = part.columns.str.strip()
        frames.append(part[["song_id", "valence_mean", "arousal_mean"]])
    df = pd.concat(frames, ignore_index=True).dropna()
    df["song_id"] = df["song_id"].astype(int)
    df = df.drop_duplicates(subset="song_id")

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