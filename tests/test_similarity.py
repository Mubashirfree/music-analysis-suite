import numpy as np

from backend.similarity import SimilarityIndex, _prepare


def make(n=60, d=768, seed=0):
    rng = np.random.default_rng(seed)
    return [f"s{i}" for i in range(n)], rng.normal(size=(n, d)).astype(np.float32)


def test_a_song_is_its_own_nearest_neighbour():
    uids, x = make()
    idx = SimilarityIndex.build(uids, x)
    assert idx.search(x[3], k=1)[0]["uid"] == "s3"


def test_search_matches_brute_force():
    uids, x = make()
    idx = SimilarityIndex.build(uids, x)
    got = [r["uid"] for r in idx.search(x[7], k=5, exclude_uid="s7")]
    xn = _prepare(x, x.mean(axis=0))
    sims = xn @ xn[7]
    sims[7] = -np.inf
    want = [uids[i] for i in np.argsort(-sims)[:5]]
    assert got == want


def test_save_and_load_roundtrip(tmp_path):
    uids, x = make()
    idx = SimilarityIndex.build(uids, x)
    idx.save(tmp_path)
    again = SimilarityIndex.load(tmp_path)
    assert again.search(x[3], k=3) == idx.search(x[3], k=3)

def test_added_song_can_be_found():
    uids, x = make()
    idx = SimilarityIndex.build(uids, x)
    new = np.random.default_rng(5).normal(size=768).astype(np.float32)
    idx.add("new_song", new)
    assert idx.search(new, k=1)[0]["uid"] == "new_song"

def test_removed_song_is_gone_and_others_keep_their_place():
    uids, x = make()
    idx = SimilarityIndex.build(uids, x)
    idx.remove("s10")
    assert "s10" not in [r["uid"] for r in idx.search(x[10], k=5)]
    assert idx.search(x[20], k=1)[0]["uid"] == "s20"