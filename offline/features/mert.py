import numpy as np
import torch
import torchaudio.functional as AF
from tqdm import tqdm
from transformers import AutoModel, Wav2Vec2FeatureExtractor

from offline.config import MERT_MODEL, MERT_SAMPLE_RATE, SAMPLE_RATE
from offline.features.cache import is_cached, load_manifest, load_waveform, save_feature
from offline.preprocessing.audio import cut_segments

FEATURE = "mert"
BATCH = 8  # segments per forward pass; lower this if the GPU runs out of memory


def load_mert(device: str):
    """Download (first run only) and load MERT and its input normaliser."""
    extractor = Wav2Vec2FeatureExtractor.from_pretrained(MERT_MODEL, trust_remote_code=True)
    model = AutoModel.from_pretrained(MERT_MODEL, trust_remote_code=True)
    model.config.output_hidden_states = True
    model = model.to(device).eval()    
    return model, extractor


@torch.no_grad()
def embed_song(y: np.ndarray, model, extractor, device: str) -> np.ndarray:
    """Return a (n_segments, 768) array: one MERT vector per 3 s segment."""
    segs = torch.from_numpy(cut_segments(y))
    segs = AF.resample(segs, SAMPLE_RATE, MERT_SAMPLE_RATE)
    vectors = []
    for i in range(0, len(segs), BATCH):
        batch = [s.numpy() for s in segs[i:i + BATCH]]
        inputs = extractor(batch, sampling_rate=MERT_SAMPLE_RATE, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        out = model(**inputs, output_hidden_states=True, return_dict=True)
        hidden = out.hidden_states
        if hidden is None:
            raise RuntimeError("MERT returned no hidden states")
        layers = torch.stack(tuple(hidden)).mean(dim=0)   # average over all layers
        vectors.append(layers.mean(dim=1).cpu())          # average over time       # average over time
    return torch.cat(vectors).numpy().astype(np.float32)


def extract_all(kind: str) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, extractor = load_mert(device)
    df = load_manifest(kind)
    for rec in tqdm(df.to_dict("records"), desc=f"mert ({kind})"):
        if is_cached(FEATURE, rec["uid"]):
            continue
        y = load_waveform(rec["proc_path"])
        save_feature(FEATURE, rec["uid"], embed_song(y, model, extractor, device))


def main() -> None:
    extract_all("mood")
    extract_all("genre")


if __name__ == "__main__":
    main()