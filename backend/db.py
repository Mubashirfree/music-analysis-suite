"""SQLite catalogue (P2.2): songs, predictions, embeddings and tags.

Every query uses ? placeholders, so file names and tags typed by a user are never
pasted into the SQL text.
"""
import json
import sqlite3

import numpy as np

from offline.config import DATA_PROCESSED

DB_PATH = DATA_PROCESSED.parent / "catalogue.db"  # data/catalogue.db (git-ignored)

SCHEMA = """
CREATE TABLE IF NOT EXISTS songs (
    song_id INTEGER PRIMARY KEY AUTOINCREMENT,
    filename TEXT NOT NULL,
    file_path TEXT NOT NULL UNIQUE,
    duration_s REAL,
    added_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS genre_predictions (
    song_id INTEGER PRIMARY KEY REFERENCES songs(song_id) ON DELETE CASCADE,
    genre TEXT NOT NULL,
    probabilities_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS mood_predictions (
    song_id INTEGER PRIMARY KEY REFERENCES songs(song_id) ON DELETE CASCADE,
    valence REAL NOT NULL,
    arousal REAL NOT NULL,
    quadrant TEXT NOT NULL,
    quadrant_probs_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS embeddings (
    song_id INTEGER PRIMARY KEY REFERENCES songs(song_id) ON DELETE CASCADE,
    vector BLOB NOT NULL
);
CREATE TABLE IF NOT EXISTS tags (
    tag_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE
);
CREATE TABLE IF NOT EXISTS song_tags (
    song_id INTEGER NOT NULL REFERENCES songs(song_id) ON DELETE CASCADE,
    tag_id INTEGER NOT NULL REFERENCES tags(tag_id) ON DELETE CASCADE,
    source TEXT NOT NULL CHECK (source IN ('auto', 'manual')),
    PRIMARY KEY (song_id, tag_id)
);
"""


def connect(path=DB_PATH):
    """Open the database (":memory:" for tests) and make sure the tables exist."""
    conn = sqlite3.connect(str(path), check_same_thread=False)        
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(SCHEMA)
    return conn


def add_song(conn, filename, file_path, duration_s=None):
    """Add a song, or find it if the same file is already there. Returns song_id."""
    conn.execute(
        "INSERT OR IGNORE INTO songs (filename, file_path, duration_s) VALUES (?, ?, ?)",
        (filename, file_path, duration_s),
    )
    conn.commit()
    return conn.execute("SELECT song_id FROM songs WHERE file_path = ?", (file_path,)).fetchone()[0]


def save_analysis(conn, song_id, result):
    """Store the dict returned by offline.inference.analyse(); replaces older results."""
    genre, mood = result["genre"], result["mood"]
    conn.execute(
        "INSERT OR REPLACE INTO genre_predictions VALUES (?, ?, ?)",
        (song_id, genre["genre"], json.dumps(genre["probabilities"])),
    )
    conn.execute(
        "INSERT OR REPLACE INTO mood_predictions VALUES (?, ?, ?, ?, ?)",
        (song_id, mood["valence"], mood["arousal"], mood["quadrant"], json.dumps(mood["quadrant_probabilities"])),
    )
    vector = np.asarray(result["embedding"], dtype=np.float32)
    conn.execute("INSERT OR REPLACE INTO embeddings VALUES (?, ?)", (song_id, vector.tobytes()))
    conn.commit()


def get_embedding(conn, song_id):
    row = conn.execute("SELECT vector FROM embeddings WHERE song_id = ?", (song_id,)).fetchone()
    return None if row is None else np.frombuffer(row["vector"], dtype=np.float32)


def add_tag(conn, song_id, name, source="auto"):
    """Attach a tag (lower-cased, trimmed) to a song; source is 'auto' or 'manual'."""
    name = name.strip().lower()
    conn.execute("INSERT OR IGNORE INTO tags (name) VALUES (?)", (name,))
    tag_id = conn.execute("SELECT tag_id FROM tags WHERE name = ?", (name,)).fetchone()[0]
    conn.execute(
        "INSERT OR IGNORE INTO song_tags (song_id, tag_id, source) VALUES (?, ?, ?)",
        (song_id, tag_id, source),
    )
    conn.commit()


def get_tags(conn, song_id):
    rows = conn.execute(
        "SELECT t.name, st.source FROM song_tags st JOIN tags t ON t.tag_id = st.tag_id "
        "WHERE st.song_id = ? ORDER BY t.name",
        (song_id,),
    ).fetchall()
    return [{"name": r["name"], "source": r["source"]} for r in rows]


def get_song(conn, song_id):
    """One song with its genre, mood and tags, or None if it does not exist."""
    s = conn.execute("SELECT * FROM songs WHERE song_id = ?", (song_id,)).fetchone()
    if s is None:
        return None
    g = conn.execute("SELECT * FROM genre_predictions WHERE song_id = ?", (song_id,)).fetchone()
    m = conn.execute("SELECT * FROM mood_predictions WHERE song_id = ?", (song_id,)).fetchone()
    return {
        "song_id": s["song_id"],
        "filename": s["filename"],
        "file_path": s["file_path"],
        "duration_s": s["duration_s"],
        "added_at": s["added_at"],
        "genre": None if g is None else {"genre": g["genre"], "probabilities": json.loads(g["probabilities_json"])},
        "mood": None if m is None else {
            "valence": m["valence"],
            "arousal": m["arousal"],
            "quadrant": m["quadrant"],
            "quadrant_probabilities": json.loads(m["quadrant_probs_json"]),
        },
        "tags": get_tags(conn, song_id),
    }


def list_songs(conn, genre=None, quadrant=None, tag=None, limit=50, offset=0):
    """Songs in id order, optionally filtered by predicted genre, quadrant or tag."""
    sql = (
        "SELECT s.song_id FROM songs s "
        "LEFT JOIN genre_predictions g ON g.song_id = s.song_id "
        "LEFT JOIN mood_predictions m ON m.song_id = s.song_id WHERE 1 = 1"
    )
    args = []
    if genre:
        sql += " AND g.genre = ?"
        args.append(genre)
    if quadrant:
        sql += " AND m.quadrant = ?"
        args.append(quadrant)
    if tag:
        sql += (" AND s.song_id IN (SELECT st.song_id FROM song_tags st "
                "JOIN tags t ON t.tag_id = st.tag_id WHERE t.name = ?)")
        args.append(tag.strip().lower())
    sql += " ORDER BY s.song_id LIMIT ? OFFSET ?"
    args += [limit, offset]
    return [get_song(conn, r["song_id"]) for r in conn.execute(sql, args).fetchall()]

def clear_auto_tags(conn, song_id):
    """Remove the model-made tags of a song; manual tags stay."""
    conn.execute("DELETE FROM song_tags WHERE song_id = ? AND source = 'auto'", (song_id,))
    conn.commit()

def delete_song(conn, song_id):
    """Delete a song; its predictions, embedding and tag links go with it (cascade)."""
    conn.execute("DELETE FROM songs WHERE song_id = ?", (song_id,))
    conn.commit()