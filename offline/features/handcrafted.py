import librosa
import numpy as np
from tqdm import tqdm

from offline.config import HOP_LENGTH, N_FFT, SAMPLE_RATE
from offline.features.cache import is_cached, load_manifest, load_waveform, save_feature

FEATURE = "handcrafted"
N_MFCC = 5  # Tzanetakis & Cook use the first five MFCCs, excluding the DC term


def spectral_flux(y: np.ndarray) -> np.ndarray:
    """Squared change between successive length-normalised magnitude spectra."""
    mag = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH))
    mag = mag / (np.linalg.norm(mag, axis=0, keepdims=True) + 1e-10)
    return np.sum(np.diff(mag, axis=1) ** 2, axis=0)


def handcrafted_summary(y: np.ndarray) -> np.ndarray:
    """Mean and std of MFCC 1-5, centroid, rolloff, ZCR, flux; low-energy fraction; tempo (20 numbers)."""
    kw = dict(sr=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH)
    mfcc = librosa.feature.mfcc(y=y, n_mfcc=N_MFCC + 1, **kw)[1:]   # drop coefficient 0
    centroid = librosa.feature.spectral_centroid(y=y, **kw)
    rolloff = librosa.feature.spectral_rolloff(y=y, **kw)
    zcr = librosa.feature.zero_crossing_rate(y, hop_length=HOP_LENGTH)
    flux = spectral_flux(y)[None, :]
    rms = librosa.feature.rms(y=y, hop_length=HOP_LENGTH)[0]
    tempo, _ = librosa.beat.beat_track(y=y, sr=SAMPLE_RATE, hop_length=HOP_LENGTH)

    n = flux.shape[1]                                                # shortest series
    rows = np.vstack([mfcc[:, :n], centroid[:, :n], rolloff[:, :n], zcr[:, :n], flux])   # (9, n)
    stats = np.concatenate([rows.mean(axis=1), rows.std(axis=1)])    # 18 numbers
    low_energy = np.mean(rms < rms.mean())                           # 1 number
    return np.concatenate([stats, [low_energy], np.atleast_1d(tempo)[:1]]).astype(np.float32)


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