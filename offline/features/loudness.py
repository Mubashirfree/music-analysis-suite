import librosa
import numpy as np
from tqdm import tqdm

from offline.config import SAMPLE_RATE
from offline.features.cache import is_cached, load_manifest, save_feature

FEATURE = "loudness"


def loudness_db(path: str) -> np.ndarray:
    """Loudness of the raw, un-normalised clip: RMS of the whole waveform in dB."""
    y, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    rms = np.sqrt(np.mean(y.astype(np.float64) ** 2))
    return np.array([20.0 * np.log10(rms + 1e-10)], dtype=np.float32)


def main() -> None:
    df = load_manifest("mood")
    for rec in tqdm(df.to_dict("records"), desc="loudness"):
        if is_cached(FEATURE, rec["uid"]):
            continue
        save_feature(FEATURE, rec["uid"], loudness_db(rec["path"]))


if __name__ == "__main__":
    main()