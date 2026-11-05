import pandas as pd
from sklearn.model_selection import train_test_split

from offline.config import SEED


def split_by_song(df: pd.DataFrame, strata: str, val_size: float = 0.15,
                  test_size: float = 0.15, seed: int = SEED) -> pd.DataFrame:
    """Add a 'split' column (train/val/test). Each row is one song, so no song is in two splits."""
    train_val, test = train_test_split(
        df, test_size=test_size, stratify=df[strata], random_state=seed
    )
    train, val = train_test_split(
        train_val, test_size=val_size / (1 - test_size),
        stratify=train_val[strata], random_state=seed,
    )
    out = df.copy()
    out["split"] = "train"
    out.loc[val.index, "split"] = "val"
    out.loc[test.index, "split"] = "test"
    return out