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
- [ ] P1.2 Dataset acquisition
- [ ] P1.3 Audio preprocessing
- [ ] P1.4 Feature extraction
- [ ] P1.5 Random Forest baseline
- [ ] P1.6 CNN with ECA attention (genre)
- [ ] P1.7 Multitask mood model
- [ ] P1.8 Cross-dataset / cross-lingual split
- [ ] P1.9 Model export