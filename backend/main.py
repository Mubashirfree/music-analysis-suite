"""FastAPI backend (P2.3).

    GET  /health                 library size and index size
    POST /songs                  upload audio -> analyse -> store -> add to the search index
    GET  /songs                  list; filter by genre, quadrant or tag
    GET  /songs/{id}             one song with genre, mood and tags
    GET  /songs/{id}/similar     nearest songs by embedding
    POST /songs/{id}/tags        add a manual tag

create_app() takes the analysis function and the storage folders as arguments, so tests
can run it with a fake analyser and temporary folders (no GPU, no models, no real audio).
Run it with:  uvicorn backend.main:app --port 8000   (then open /docs in a browser)
"""
import threading
import uuid
from functools import lru_cache
from pathlib import Path

from fastapi import Depends, FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from backend import db
from backend.similarity import INDEX_DIR, SimilarityIndex
from backend.autotag import apply_auto_tags
from offline.config import DATA_PROCESSED
from offline.features.cache import load_manifest

UPLOAD_DIR = DATA_PROCESSED.parent / "uploads"  # data/uploads (git-ignored)
AUDIO_TYPES = {".wav": "audio/wav", ".mp3": "audio/mpeg", ".flac": "audio/flac", ".ogg": "audio/ogg"}
ALLOWED = {".wav", ".mp3", ".flac", ".ogg"}
MAX_BYTES = 50 * 1024 * 1024  # 50 MB per upload


class TagIn(BaseModel):
    name: str = Field(min_length=1, max_length=40)


@lru_cache(maxsize=1)
def _library_info():
    """uid -> dataset and label, read from the manifests (empty if they are missing)."""
    info = {}
    try:
        for row in load_manifest("genre").itertuples():
            info[row.uid] = {"dataset": row.dataset, "genre": row.genre}
        for row in load_manifest("mood").itertuples():
            info[row.uid] = {"dataset": row.dataset, "mood": row.mood}
    except Exception:
        return {}
    return info


def _duration(path):
    try:
        import librosa

        return round(float(librosa.get_duration(path=str(path))), 2)
    except Exception:
        return None

def _public(song):
    """What a client may see: everything except the server's file path."""
    if song is None:
        return None
    return {k: v for k, v in song.items() if k != "file_path"}

def _describe(conn, hit):
    """Turn a raw search hit into something a client can show."""
    uid = hit["uid"]
    if uid.startswith("song_"):
        return {"source": "catalogue", "score": hit["score"], "song": _public(db.get_song(conn, int(uid[5:])))}
    return {"source": "library", "uid": uid, "score": hit["score"], "info": _library_info().get(uid, {})}


def create_app(analyse_fn=None, db_path=db.DB_PATH, index_dir=INDEX_DIR, upload_dir=UPLOAD_DIR):
    app = FastAPI(title="Music Analysis Suite")
    # The React dev server (Phase 3) runs on another port, so the browser needs permission.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:5173", "http://127.0.0.1:5173",
            "http://localhost:3000", "http://127.0.0.1:3000",
        ],        
        allow_methods=["*"],
        allow_headers=["*"],
    )
    gpu_lock = threading.Lock()    # one analysis at a time (MERT needs the GPU memory)
    index_lock = threading.RLock()  # the index is changed by uploads and read by searches
    state = {"index": None}

    def analyse(path):
        if analyse_fn is not None:
            return analyse_fn(path)
        from offline.inference import analyse as real_analyse  # loads the models on first use

        return real_analyse(path)

    def get_index():
        with index_lock:
            if state["index"] is None:
                try:
                    state["index"] = SimilarityIndex.load(index_dir)
                except Exception:
                    raise HTTPException(503, "Similarity index not found. Run: python -m backend.similarity")
            return state["index"]

    def get_conn():
        conn = db.connect(db_path)
        try:
            yield conn
        finally:
            conn.close()

    @app.get("/health")
    def health(conn=Depends(get_conn)):
        songs = conn.execute("SELECT COUNT(*) FROM songs").fetchone()[0]
        return {"status": "ok", "songs": songs, "indexed": get_index().index.ntotal}

    @app.post("/songs", status_code=201)
    def upload_song(file: UploadFile = File(...), conn=Depends(get_conn)):
        suffix = Path(file.filename or "").suffix.lower()
        if suffix not in ALLOWED:
            raise HTTPException(415, f"Unsupported file type. Use one of: {sorted(ALLOWED)}")
        get_index()  # fail early if the index is missing
        upload_dir.mkdir(parents=True, exist_ok=True)
        dest = upload_dir / f"{uuid.uuid4().hex}{suffix}"  # never use the client's file name as a path
        try:
            size = 0
            with open(dest, "wb") as out:
                while chunk := file.file.read(1024 * 1024):
                    size += len(chunk)
                    if size > MAX_BYTES:
                        raise HTTPException(413, "File too large (limit 50 MB).")
                    out.write(chunk)
            if size == 0:
                raise HTTPException(422, "The file is empty.")
            try:
                with gpu_lock:
                    result = analyse(str(dest))
            except Exception as exc:
                raise HTTPException(422, "Could not analyse this audio file.") from exc
        except Exception:
            dest.unlink(missing_ok=True)  # leave nothing behind when anything fails
            raise
        song_id = db.add_song(conn, Path(file.filename).name, str(dest), _duration(dest))
        db.save_analysis(conn, song_id, result)
        apply_auto_tags(conn, song_id, result)
        with index_lock:
            index = get_index()
            index.add(f"song_{song_id}", result["embedding"])
            index.save(index_dir)
        return _public(db.get_song(conn, song_id))

    @app.get("/songs")
    def list_songs(
        genre: str | None = None,
        quadrant: str | None = None,
        tag: str | None = None,
        limit: int = Query(50, ge=1, le=200),
        offset: int = Query(0, ge=0),
        conn=Depends(get_conn),
    ):
        return [_public(s) for s in db.list_songs(conn, genre, quadrant, tag, limit, offset)]
   
    @app.get("/songs/{song_id}")
    def get_song(song_id: int, conn=Depends(get_conn)):
        song = db.get_song(conn, song_id)
        if song is None:
            raise HTTPException(404, "Song not found.")
        return _public(song)

    @app.get("/songs/{song_id}/audio")
    def song_audio(song_id: int, conn=Depends(get_conn)):
        song = db.get_song(conn, song_id)
        if song is None:
            raise HTTPException(404, "Song not found.")
        path = Path(song["file_path"])  # from the database, never from the request
        if not path.is_file():
            raise HTTPException(404, "The audio file is no longer on the server.")
        media_type = AUDIO_TYPES.get(path.suffix.lower(), "application/octet-stream")
        return FileResponse(path, media_type=media_type)
   
    @app.get("/songs/{song_id}/similar")
    def similar(song_id: int, k: int = Query(10, ge=1, le=50), conn=Depends(get_conn)):
        vector = db.get_embedding(conn, song_id)
        if vector is None:
            raise HTTPException(404, "Song not found.")
        with index_lock:
            hits = get_index().search(vector, k=k, exclude_uid=f"song_{song_id}")
        return [_describe(conn, h) for h in hits]

    @app.post("/songs/{song_id}/tags", status_code=201)
    def tag_song(song_id: int, body: TagIn, conn=Depends(get_conn)):
        if db.get_song(conn, song_id) is None:
            raise HTTPException(404, "Song not found.")
        name = body.name.strip()
        if not name:
            raise HTTPException(422, "The tag name is empty.")
        db.add_tag(conn, song_id, name, "manual")
        return db.get_tags(conn, song_id)

    @app.delete("/songs/{song_id}", status_code=204)
    def delete_song(song_id: int, conn=Depends(get_conn)):
        song = db.get_song(conn, song_id)
        if song is None:
            raise HTTPException(404, "Song not found.")
        db.delete_song(conn, song_id)
        with index_lock:
            index = get_index()
            if f"song_{song_id}" in index.uids:
                index.remove(f"song_{song_id}")
                index.save(index_dir)
        path = Path(song["file_path"])
        # Only files the API itself saved are deleted; imported originals are never touched.
        if path.parent.resolve() == Path(upload_dir).resolve():
            path.unlink(missing_ok=True)

    return app


app = create_app()