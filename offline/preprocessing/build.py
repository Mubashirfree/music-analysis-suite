import numpy as np
import pandas as pd
from tqdm import tqdm

from offline.config import DATA_PROCESSED
from offline.data.gtzan import load_gtzan
from offline.data.mood import add_mood_category, load_mood
from offline.preprocessing.audio import preprocess, segment_starts
from offline.preprocessing.splits import split_by_song


def process_table(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess every song once, save it as .npy, and record length and segment count."""
    rows = []
    failed = []
    for rec in tqdm(df.to_dict("records"), desc="Preprocessing"):
        out_path = DATA_PROCESSED / rec["dataset"] / f"{rec['uid']}.npy"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            if out_path.exists():
                n_samples = len(np.load(out_path, mmap_mode="r"))
            else:
                y = preprocess(rec["path"])
                np.save(out_path, y.astype(np.float16))
                n_samples = len(y)
        except Exception as e:
            failed.append((rec["uid"], str(e)[:80]))
            continue
        rec["proc_path"] = str(out_path)
        rec["n_samples"] = n_samples
        rec["n_segments"] = len(segment_starts(n_samples))
        rows.append(rec)
    for uid, msg in failed:
        print(f"FAILED {uid}: {msg}")
    return pd.DataFrame(rows)


def build_genre() -> pd.DataFrame:
    df = load_gtzan()
    df["dataset"] = "gtzan"
    df["uid"] = "gtzan_" + df["filename"].str.replace(".wav", "", regex=False)
    df = split_by_song(df, strata="genre")
    return process_table(df)


def build_mood() -> pd.DataFrame:
    df = add_mood_category(load_mood())
    df["uid"] = df["dataset"] + "_" + df["song_id"].astype(str)
    df["strata"] = df["dataset"] + "_" + df["mood"]
    df = split_by_song(df, strata="strata")
    return process_table(df)


def summarise(name: str, df: pd.DataFrame, group: str) -> None:
    print(f"\n{name}: {len(df)} songs, {int(df['n_segments'].sum())} segments")
    print(pd.crosstab(df[group], df["split"]))


def main() -> None:
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    genre = build_genre()
    genre.to_csv(DATA_PROCESSED / "genre_manifest.csv", index=False)
    summarise("Genre (GTZAN)", genre, "genre")

    mood = build_mood()
    mood.to_csv(DATA_PROCESSED / "mood_manifest.csv", index=False)
    summarise("Mood (DEAM + PMEmo)", mood, "mood")


if __name__ == "__main__":
    main()