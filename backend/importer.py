"""Batch import of a folder of audio files (P2.4).

    python -m backend.importer <folder> [max_songs]

Every supported file under the folder is analysed, stored in the catalogue, tagged
automatically and added to the search index. Files already imported are skipped, so the
command can be repeated safely. This is a command, not an API endpoint, on purpose:
an endpoint that scans server folders would expose the server's files.
Stop the API server first: it keeps its own copy of the index and would overwrite this one.
"""
import json
import sys
from pathlib import Path

from backend import db
from backend.autotag import apply_auto_tags
from backend.main import ALLOWED, _duration
from backend.similarity import INDEX_DIR, SimilarityIndex


def import_folder(conn, folder, analyse_fn, index, index_dir, limit=None):
    files = sorted(p.resolve() for p in Path(folder).rglob("*") if p.is_file() and p.suffix.lower() in ALLOWED)
    known = {row[0] for row in conn.execute("SELECT file_path FROM songs")}
    summary = {"imported": 0, "skipped": 0, "failed": []}
    for path in files:
        if limit is not None and summary["imported"] >= limit:
            break
        if str(path) in known:
            summary["skipped"] += 1
            continue
        try:
            result = analyse_fn(str(path))
        except Exception as exc:  # one bad file must not stop the whole import
            summary["failed"].append({"file": str(path), "error": str(exc)[:200]})
            continue
        song_id = db.add_song(conn, path.name, str(path), _duration(path))
        db.save_analysis(conn, song_id, result)
        apply_auto_tags(conn, song_id, result)
        index.add(f"song_{song_id}", result["embedding"])
        summary["imported"] += 1
        if summary["imported"] % 10 == 0:
            print(f"  imported {summary['imported']} songs", flush=True)
    index.save(index_dir)
    return summary


def main():
    if len(sys.argv) < 2:
        print("usage: python -m backend.importer <folder> [max_songs]")
        return 1
    folder = Path(sys.argv[1]).expanduser()
    if not folder.is_dir():
        print(f"Not a folder: {folder}")
        return 1
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else None
    from offline.inference import analyse  # loads the models

    summary = import_folder(db.connect(), folder, analyse, SimilarityIndex.load(), INDEX_DIR, limit)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())