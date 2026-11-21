"""Turn a song's predictions into tags (P2.4).

Works on the dict from offline.inference.analyse() and also on a stored song from
backend.db.get_song(), since both have the same genre/mood keys. Low-confidence
predictions are not written as tags, so a guess is never stored as a fact.
"""
from backend import db

GENRE_MIN = 0.40      # smallest top-genre probability that earns a genre tag
QUADRANT_MIN = 0.50   # smallest quadrant probability that earns a mood tag
HIGH, LOW = 0.60, 0.40  # checked on the library (P2.6): tags 14% to 33% of songs, see docs/results/tag_calibration.txt

def tags_for(result):
    tags = []
    genre = result["genre"]
    if genre["probabilities"][genre["genre"]] >= GENRE_MIN:
        tags.append(genre["genre"])
    mood = result["mood"]
    if mood["quadrant_probabilities"][mood["quadrant"]] >= QUADRANT_MIN:
        tags.append(mood["quadrant"].lower())
    if mood["arousal"] >= HIGH:
        tags.append("energetic")
    elif mood["arousal"] <= LOW:
        tags.append("calm")
    if mood["valence"] >= HIGH:
        tags.append("positive")
    elif mood["valence"] <= LOW:
        tags.append("negative")
    return tags


def apply_auto_tags(conn, song_id, result):
    """Replace the song's auto tags with the ones its predictions earn; return them."""
    db.clear_auto_tags(conn, song_id)
    tags = tags_for(result)
    for name in tags:
        db.add_tag(conn, song_id, name, "auto")
    return tags