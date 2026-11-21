import numpy as np

from backend.db import add_song, add_tag, connect, get_embedding, get_song, list_songs, save_analysis


def fake_result(genre="jazz", quadrant="Sad/Depressed"):
    return {
        "genre": {"genre": genre, "probabilities": {genre: 0.9, "rock": 0.1}},
        "mood": {"valence": 0.4, "arousal": 0.3, "quadrant": quadrant, "quadrant_probabilities": {quadrant: 1.0}},
        "embedding": [0.5] * 768,
    }


def test_add_song_is_idempotent():
    conn = connect(":memory:")
    first = add_song(conn, "a.mp3", "/music/a.mp3")
    second = add_song(conn, "a.mp3", "/music/a.mp3")
    assert first == second
    assert get_song(conn, 999) is None


def test_analysis_round_trip():
    conn = connect(":memory:")
    sid = add_song(conn, "a.mp3", "/music/a.mp3", 30.0)
    save_analysis(conn, sid, fake_result())
    song = get_song(conn, sid)
    assert song["genre"]["genre"] == "jazz"
    assert song["mood"]["valence"] == 0.4
    assert np.allclose(get_embedding(conn, sid), 0.5)


def test_tags_and_filters():
    conn = connect(":memory:")
    s1 = add_song(conn, "a.mp3", "/m/a.mp3")
    s2 = add_song(conn, "b.mp3", "/m/b.mp3")
    save_analysis(conn, s1, fake_result("jazz"))
    save_analysis(conn, s2, fake_result("rock", "Happy/Excited"))
    add_tag(conn, s1, "Chill ", "auto")
    assert [s["song_id"] for s in list_songs(conn, genre="rock")] == [s2]
    assert [s["song_id"] for s in list_songs(conn, tag="chill")] == [s1]
    assert get_song(conn, s1)["tags"] == [{"name": "chill", "source": "auto"}]