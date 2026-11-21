"""Similarity search over song embeddings (P2.1).

Library = every song in both manifests: 999 GTZAN + 1802 DEAM + 767 PMEmo.
Vector per song = mean MERT embedding of the whole clip, the same recipe as
offline.inference.embed(). Mood songs reuse their cached MERT vectors (identical
to embed()); GTZAN songs are embedded from the raw files, because the cached
genre vectors came from trimmed 30 s crops.

Search: subtract the library mean, scale each vector to length 1, then the inner
product is the cosine similarity. Exact search (IndexFlatIP) is instant at this size.
"""
import json
import sys

import faiss
import numpy as np

from offline.config import MODELS_DIR
from offline.features.cache import is_cached, load_feature, load_manifest, save_feature

INDEX_DIR = MODELS_DIR / "similarity"
VEC = "mert_full"  # cache folder: one 768-number vector per song


def library_vectors():
    """Return (uids, matrix) for all songs, computing and caching what is missing."""
    from offline.inference import embed  # imported here so MERT only loads when needed

    uids, vecs = [], []
    for kind in ("genre", "mood"):
        df = load_manifest(kind)
        for i, row in enumerate(df.itertuples(), 1):
            if not is_cached(VEC, row.uid):
                if kind == "mood":
                    v = load_feature("mert", row.uid).mean(axis=0)
                else:
                    v = np.array(embed(row.path), dtype=np.float32)
                save_feature(VEC, row.uid, v.astype(np.float32))
            uids.append(row.uid)
            vecs.append(load_feature(VEC, row.uid))
            if i % 250 == 0:
                print(f"  {kind}: {i}/{len(df)}", flush=True)
    return uids, np.stack(vecs).astype(np.float32)


def _prepare(x, mean):
    """Centre on the library mean, then scale every row to length 1."""
    x = (x - mean).astype(np.float32)
    x /= np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return np.ascontiguousarray(x)


class SimilarityIndex:
    def __init__(self, index, uids, mean):
        self.index, self.uids, self.mean = index, uids, mean

    @classmethod
    def build(cls, uids, vectors):
        mean = vectors.mean(axis=0)
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(_prepare(vectors, mean))
        return cls(index, list(uids), mean)

    def add(self, uid, vector):
        """Add one new song to the index, using the stored library mean."""
        self.index.add(_prepare(np.asarray(vector, dtype=np.float32)[None, :], self.mean))
        self.uids.append(uid)
    
    def remove(self, uid):
        """Remove one song from the index; the songs after it shift up by one row."""
        row = self.uids.index(uid)
        self.index.remove_ids(np.array([row], dtype=np.int64))
        del self.uids[row]
   
    def search(self, vector, k=10, exclude_uid=None):
        """Nearest songs to one 768-number vector, best first."""
        q = _prepare(np.asarray(vector, dtype=np.float32)[None, :], self.mean)
        scores, rows = self.index.search(q, k + 1 if exclude_uid else k)
        out = []
        for s, r in zip(scores[0], rows[0]):
            uid = self.uids[r]
            if uid != exclude_uid:
                out.append({"uid": uid, "score": round(float(s), 4)})
        return out[:k]

    def save(self, folder=INDEX_DIR):
        folder.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(folder / "index.faiss"))
        (folder / "uids.json").write_text(json.dumps(self.uids))
        np.save(folder / "mean.npy", self.mean)

    @classmethod
    def load(cls, folder=INDEX_DIR):
        index = faiss.read_index(str(folder / "index.faiss"))
        uids = json.loads((folder / "uids.json").read_text())
        return cls(index, uids, np.load(folder / "mean.npy"))


def main():
    print("Collecting song vectors (GTZAN needs MERT, so the first run is slow)...")
    uids, vectors = library_vectors()
    index = SimilarityIndex.build(uids, vectors)
    index.save()
    print(f"Index built: {len(uids)} songs, saved to {INDEX_DIR}")
    probe = "deam_2"
    print(f"Nearest songs to {probe}:")
    for hit in index.search(vectors[uids.index(probe)], k=5, exclude_uid=probe):
        print("  ", hit)


if __name__ == "__main__":
    sys.exit(main())