import librosa
import numpy as np
from tqdm import tqdm

from offline.config import HOP_LENGTH, N_FFT, SAMPLE_RATE
from offline.features.cache import is_cached, load_manifest, load_waveform, save_feature

FEATURE = "chroma"


def chroma_summary(y: np.ndarray) -> np.ndarray:
    """12 pitch-class means followed by 12 standard deviations (24 numbers)."""
    chroma = librosa.feature.chroma_stft(y=y, sr=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH)
    return np.concatenate([chroma.mean(axis=1), chroma.std(axis=1)]).astype(np.float32)


def extract_all(kind: str) -> None:
    df = load_manifest(kind)
    for rec in tqdm(df.to_dict("records"), desc=f"chroma ({kind})"):
        if is_cached(FEATURE, rec["uid"]):
            continue
        save_feature(FEATURE, rec["uid"], chroma_summary(load_waveform(rec["proc_path"])))


def main() -> None:
    extract_all("genre")
    extract_all("mood")


if __name__ == "__main__":
    main()