import librosa
import numpy as np
from tqdm import tqdm

from offline.config import HOP_LENGTH, N_FFT, N_MELS, SAMPLE_RATE
from offline.features.cache import is_cached, load_manifest, load_waveform, save_feature
from offline.preprocessing.audio import cut_segments

FEATURE = "logmel"


def logmel_segments(y: np.ndarray) -> np.ndarray:
    """Cut a song into segments and return log-mel images, shape (n_seg, n_mels, frames)."""
    out = []
    for seg in cut_segments(y):
        mel = librosa.feature.melspectrogram(
            y=seg, sr=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS
        )
        out.append(librosa.power_to_db(mel, ref=1.0, top_db=80.0))
    return np.stack(out).astype(np.float16)


def extract_all(kind: str) -> None:
    df = load_manifest(kind)
    for rec in tqdm(df.to_dict("records"), desc=f"logmel ({kind})"):
        if is_cached(FEATURE, rec["uid"]):
            continue
        save_feature(FEATURE, rec["uid"], logmel_segments(load_waveform(rec["proc_path"])))


def main() -> None:
    extract_all("genre")


if __name__ == "__main__":
    main()