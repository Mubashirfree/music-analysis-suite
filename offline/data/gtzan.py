from pathlib import Path

import pandas as pd

from offline.config import DATA_RAW, GENRES

GTZAN_DIR = DATA_RAW / "gtzan"

# Known corrupted file in many GTZAN copies; excluded from the table.
BAD_FILES = {"jazz.00054.wav"}


def load_gtzan(root: Path = GTZAN_DIR) -> pd.DataFrame:
    """Scan the GTZAN folder and return a table of audio files and genre labels."""
    wavs = sorted(root.rglob("*.wav"))
    if not wavs:
        raise FileNotFoundError(f"No .wav files found under {root}")

    rows = []
    for path in wavs:
        genre = path.parent.name
        if genre not in GENRES or path.name in BAD_FILES:
            continue
        rows.append({
            "path": str(path),
            "filename": path.name,
            "genre": genre,
            "label": GENRES.index(genre),
        })

    return pd.DataFrame(rows)


def main() -> None:
    df = load_gtzan()
    print(f"Loaded {len(df)} GTZAN files")
    print(df["genre"].value_counts().sort_index())
    print(df.head())


if __name__ == "__main__":
    main()