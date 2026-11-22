from fastapi.testclient import TestClient

from backend import db
from backend.main import create_app


def test_audio_is_served_and_missing_cases_give_404(tmp_path):
    db_path = tmp_path / "test.db"
    audio = tmp_path / "clip.wav"
    audio.write_bytes(b"RIFFdemo")

    conn = db.connect(db_path)
    song_id = db.add_song(conn, "clip.wav", str(audio), 1.0)
    conn.close()

    client = TestClient(create_app(db_path=db_path, index_dir=tmp_path / "idx", upload_dir=tmp_path / "up"))

    response = client.get(f"/songs/{song_id}/audio")
    assert response.status_code == 200
    assert response.content == b"RIFFdemo"
    assert response.headers["content-type"].startswith("audio/")

    assert client.get("/songs/999/audio").status_code == 404  # no such song

    audio.unlink()
    assert client.get(f"/songs/{song_id}/audio").status_code == 404  # file gone