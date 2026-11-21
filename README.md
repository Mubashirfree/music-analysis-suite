# Music Analysis Suite

Genre and Mood Based Personal Music Cataloguing Using Multitask Deep Learning.

**Team:** Syed Mubashir Ahmed (1RV25MC106), Nitin S N (1RV25MC063)

## Project structure

| Folder | Purpose | Phase |
|---|---|---|
| `offline/` | data prep, feature extraction, model training | 1 |
| `models/` | exported model checkpoints | 1 |
| `backend/` | FastAPI, SQLite, FAISS | 2 |
| `frontend/` | React (Vite) app | 3 |
| `docs/` | notes, figures, results | all |

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install torch torchaudio
python -m offline.check_env
```

## Progress

- [x] P1.1 Repository structure, environment, config
- [x] P1.2 Dataset acquisition
- [x] P1.3 Audio preprocessing
- [x] P1.4 Feature extraction
- [x] P1.5 Random Forest baseline
- [x] P1.6 CNN with ECA attention (genre)
- [x] P1.7 Multitask mood model
- [x] P1.8 Cross-dataset / cross-lingual split
- [x] P1.9 Model export

## Backend API (Phase 2)

Start: `uvicorn backend.main:app --port 8000`, then open http://127.0.0.1:8000/docs

| Method | Path | What it does |
|---|---|---|
| GET | /health | Songs in the catalogue and in the search index |
| POST | /songs | Upload audio (wav, mp3, flac, ogg; 50 MB max): analyse, tag, store, index |
| GET | /songs | List songs; filter with genre, quadrant or tag |
| GET | /songs/{id} | One song with genre, mood and tags |
| GET | /songs/{id}/similar | Nearest songs (library and catalogue) |
| POST | /songs/{id}/tags | Add a manual tag |
| DELETE | /songs/{id} | Delete a song, its saved file and its index entry |

Batch import (stop the server first): `python -m backend.importer <folder> [max_songs]`
Rebuild the search index: `python -m backend.similarity`