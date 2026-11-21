from backend.autotag import tags_for


def result(genre="jazz", p=0.9, quadrant="Sad/Depressed", q=0.7, v=0.3, a=0.3):
    return {
        "genre": {"genre": genre, "probabilities": {genre: p}},
        "mood": {"valence": v, "arousal": a, "quadrant": quadrant, "quadrant_probabilities": {quadrant: q}},
    }


def test_confident_prediction_gets_all_tags():
    assert tags_for(result()) == ["jazz", "sad/depressed", "calm", "negative"]


def test_low_confidence_gets_no_genre_or_mood_tag():
    assert tags_for(result(p=0.25, q=0.4, v=0.5, a=0.5)) == []


def test_energetic_positive_song():
    tags = tags_for(result("rock", 0.8, "Happy/Excited", 0.9, v=0.7, a=0.8))
    assert tags == ["rock", "happy/excited", "energetic", "positive"]