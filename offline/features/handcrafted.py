import librosa
import numpy as np
from tqdm import tqdm

from offline.config import HOP_LENGTH, N_FFT, SAMPLE_RATE
from offline.features.cache import is_cached, load_manifest, load_waveform, save_feature

FEATURE = "handcrafted"
N_MFCC = 20


def handcrafted_summary(y: np.ndarray) -> np.ndarray:
    """Mean and std of MFCC, spectral centroid, rolloff, ZCR and RMS, plus tempo (49 numbers)."""
    kw = dict(sr=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH)
    mfcc = librosa.feature.mfcc(y=y, n_mfcc=N_MFCC, **kw)
    centroid = librosa.feature.spectral_centroid(y=y, **kw)
    rolloff = librosa.feature.spectral_rolloff(y=y, **kw)
    zcr = librosa.feature.zero_crossing_rate(y, hop_length=HOP_LENGTH)
    rms = librosa.feature.rms(y=y, hop_length=HOP_LENGTH)
    tempo, _ = librosa.beat.beat_track(y=y, sr=SAMPLE_RATE, hop_length=HOP_LENGTH)

    rows = np.vstack([mfcc, centroid, rolloff, zcr, rms])        # (24, time)
    stats = np.concatenate([rows.mean(axis=1), rows.std(axis=1)])  # 48 numbers
    return np.concatenate([stats, np.atleast_1d(tempo)[:1]]).astype(np.float32)


def extract_all(kind: str) -> None:
    df = load_manifest(kind)
    for rec in tqdm(df.to_dict("records"), desc=f"handcrafted ({kind})"):
        if is_cached(FEATURE, rec["uid"]):
            continue
        save_feature(FEATURE, rec["uid"], handcrafted_summary(load_waveform(rec["proc_path"])))


def main() -> None:
    extract_all("genre")


if __name__ == "__main__":
    main()