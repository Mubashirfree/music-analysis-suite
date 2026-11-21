import io

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from backend.main import create_app
from backend.similarity import SimilarityIndex


def make_analyse():
    calls = {"n": 0}

    def fake(path):
        calls["n"] += 1
        vec = np.random.default_rng(calls["n"]).normal(size=768)
        return {
            "genre": {"genre": "jazz", "probabilities": {"jazz": 0.9, "rock": 0.1}},
            "mood": {"valence": 0.4, "arousal": 0.3, "quadrant": "Sad/Depressed",
                     "quadrant_probabilities": {"Sad/Depressed": 1.0}},
            "embedding": [float(v) for v in vec],
        }

    return fake


def build_client(tmp_path, analyse_fn):
    rng = np.random.default_rng(0)
    lib = SimilarityIndex.build([f"lib_{i}" for i in range(20)], rng.normal(size=(20, 768)).astype(np.float32))
    lib.save(tmp_path / "index")
    app = create_app(analyse_fn=analyse_fn, db_path=tmp_path / "test.db",
                     index_dir=tmp_path / "index", upload_dir=tmp_path / "uploads")
    return TestClient(app)


@pytest.fixture
def client(tmp_path):
    return build_client(tmp_path, make_analyse())


def wav_bytes():
    buf = io.BytesIO()
    sf.write(buf, np.zeros(22050, dtype="float32"), 22050, format="WAV")
    return buf.getvalue()


def upload(client, name="my song.wav"):
    return client.post("/songs", files={"file": (name, wav_bytes(), "audio/wav")})


def test_health_reports_library_size(client):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["indexed"] == 20


def test_upload_analyses_and_stores_the_song(client, tmp_path):
    resp = upload(client)
    assert resp.status_code == 201
    data = resp.json()
    assert {t["name"] for t in data["tags"]} == {"jazz", "sad/depressed", "calm", "negative"}
    assert all(t["source"] == "auto" for t in data["tags"])
    assert data["genre"]["genre"] == "jazz" and data["filename"] == "my song.wav"
    saved = list((tmp_path / "uploads").iterdir())
    assert len(saved) == 1 and saved[0].name != "my song.wav" and saved[0].suffix == ".wav"
    assert client.get(f"/songs/{data['song_id']}").json()["song_id"] == data["song_id"]
    assert len(client.get("/songs", params={"genre": "jazz"}).json()) == 1
    assert client.get("/health").json()["indexed"] == 21


def test_rejects_unsupported_file_type(client):
    resp = client.post("/songs", files={"file": ("notes.txt", b"hello", "text/plain")})
    assert resp.status_code == 415


def test_stored_name_cannot_escape_upload_folder(client, tmp_path):
    data = upload(client, "../../evil.wav").json()
    assert data["filename"] == "evil.wav"
    assert len(list((tmp_path / "uploads").iterdir())) == 1


def test_similar_songs_excludes_itself(client):
    upload(client, "a.wav")
    upload(client, "b.wav")
    hits = client.get("/songs/1/similar", params={"k": 3}).json()
    assert len(hits) == 3
    for hit in hits:
        if hit["source"] == "catalogue":
            assert hit["song"]["song_id"] != 1


def test_tags(client):
    sid = upload(client).json()["song_id"]
    ok = client.post(f"/songs/{sid}/tags", json={"name": "Late Night"})
    assert ok.status_code == 201
    tags = {(t["name"], t["source"]) for t in ok.json()}
    assert ("late night", "manual") in tags   # the manual tag is cleaned and stored
    assert ("jazz", "auto") in tags           # the auto tags from the upload are still there    assert client.post(f"/songs/{sid}/tags", json={"name": "   "}).status_code == 422
    assert client.post("/songs/999/tags", json={"name": "x"}).status_code == 404


def test_unreadable_audio_returns_422_and_leaves_nothing_behind(tmp_path):
    def broken(path):
        raise ValueError("cannot decode")

    client = build_client(tmp_path, broken)
    assert upload(client).status_code == 422
    assert list((tmp_path / "uploads").iterdir()) == []
    assert client.get("/songs").json() == []

def test_file_path_is_not_exposed(client):
    data = upload(client).json()
    assert "file_path" not in data
    assert all("file_path" not in s for s in client.get("/songs").json())
    assert "file_path" not in client.get(f"/songs/{data['song_id']}").json()


def test_delete_removes_song_file_and_index_entry(client, tmp_path):
    sid = upload(client).json()["song_id"]
    assert client.get("/health").json()["indexed"] == 21
    assert client.delete(f"/songs/{sid}").status_code == 204
    assert client.get(f"/songs/{sid}").status_code == 404
    assert list((tmp_path / "uploads").iterdir()) == []
    assert client.get("/health").json()["indexed"] == 20
    assert client.delete(f"/songs/{sid}").status_code == 404