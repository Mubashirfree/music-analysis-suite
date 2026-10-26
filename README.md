# Music Analysis Suite
Genre and Mood Based Personal Music Cataloguing Using Multitask Deep Learning.

**Team:** Syed Mubashir Ahmed (1RV25MC106), Nitin S N (1RV25MC063)

## Structure
- `offline/`  data prep, feature extraction, model training (Phase 1)
- `models/`   exported model checkpoints (Phase 1 output)
- `backend/`  FastAPI + SQLite + FAISS (Phase 2)
- `frontend/` React (Vite) app (Phase 3)
- `docs/`     architecture notes, results

## Setup
    python3 -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt

## Status
- [x] Phase 1 started: environment and repo setup
