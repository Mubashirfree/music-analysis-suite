import numpy as np
import soundfile as sf

from backend import db
from backend.importer import import_folder
from backend.similarity import SimilarityIndex


def fake_analyse(path):
    vec = np.random.default_rng(abs(hash(path)) % 1000).normal(size=768)
    return {
        "genre": {"genre": "jazz", "probabilities": {"jazz": 0.9}},
        "mood": {"valence": 0.4, "arousal": 0.3, "quadrant": "Sad/Depressed",
                 "quadrant_probabilities": {"Sad/Depressed": 1.0}},
        "embedding": [float(v) for v in vec],
    }


def write_wav(path):
    sf.write(str(path), np.zeros(2205, dtype="float32"), 22050)


def small_index():
    rng = np.random.default_rng(0)
    return SimilarityIndex.build([f"lib_{i}" for i in range(5)], rng.normal(size=(5, 768)).astype(np.float32))


def test_import_folder_adds_tags_and_skips_repeats(tmp_path):
    music = tmp_path / "music"
    (music / "sub").mkdir(parents=True)
    write_wav(music / "a.wav")
    write_wav(music / "sub" / "b.wav")
    (music / "notes.txt").write_text("not audio")
    conn = db.connect(tmp_path / "t.db")
    index = small_index()
    first = import_folder(conn, music, fake_analyse, index, tmp_path / "idx")
    assert first == {"imported": 2, "skipped": 0, "failed": []}
    assert index.index.ntotal == 7
    assert len(db.list_songs(conn)) == 2
    assert db.get_song(conn, 1)["tags"]
    again = import_folder(conn, music, fake_analyse, index, tmp_path / "idx")
    assert again["imported"] == 0 and again["skipped"] == 2


def test_import_reports_failures_and_keeps_going(tmp_path):
    music = tmp_path / "music"
    music.mkdir()
    write_wav(music / "a.wav")

    def broken(path):
        raise ValueError("cannot decode")

    conn = db.connect(tmp_path / "t.db")
    summary = import_folder(conn, music, broken, small_index(), tmp_path / "idx")
    assert summary["imported"] == 0 and len(summary["failed"]) == 1
    assert db.list_songs(conn) == []