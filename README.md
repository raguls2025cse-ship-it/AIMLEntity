# EntitySync AI — Intelligent Business Entity Resolution

> **AIML Hackathon** · Macro F0.5: 98.98% · Precision: 99.53% · Recall: 97.64%

## Overview

EntitySync AI identifies records from independent business data sources that represent the same real-world business using a trained ML model. It resolves "Acme Technologies Inc.", "ACME TECHNOLOGIES INC", and "Acme Tech Intl" to a single canonical entity — without any external APIs or business databases.

## Architecture

```
Browser (React/Vite)
        ↓
FastAPI Backend  (backend/app/main.py)
        ↓
ml_service.py  (adapter)
        ↓
src/preprocessing.py  →  src/blocking.py  →  src/features.py
        ↓
models/improved/entity_matcher.pkl  (HistGradientBoosting)
        ↓
backend/outputs/matching_results.tsv
backend/outputs/candidate_pairs.tsv
```

## Quick Start

### 1 — Backend

```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

> On Windows, run from the **`backend/`** directory. The PYTHONPATH is auto-resolved via `app/config.py`.

### 2 — Frontend

```bash
cd frontend
npm install
npm run dev
```

Open **http://localhost:5173**

---

## Project Structure

```
EntitySync/
├── frontend/                   # React + Vite UI
│   ├── src/
│   │   ├── components/
│   │   │   ├── Navbar.jsx
│   │   │   ├── UploadZone.jsx
│   │   │   ├── ProgressPanel.jsx
│   │   │   └── MatchDetailModal.jsx
│   │   ├── pages/
│   │   │   ├── Dashboard.jsx
│   │   │   ├── UploadPage.jsx
│   │   │   ├── MatchingPage.jsx
│   │   │   ├── ResultsPage.jsx
│   │   │   └── AboutPage.jsx
│   │   ├── services/api.js
│   │   ├── App.jsx
│   │   └── index.css
│   ├── .env
│   └── package.json
│
├── backend/
│   ├── app/
│   │   ├── main.py             # FastAPI app + CORS
│   │   ├── config.py           # All paths & settings
│   │   ├── routes/
│   │   │   ├── health.py       # GET /api/health
│   │   │   ├── upload.py       # POST /api/upload
│   │   │   ├── matching.py     # POST /api/match, GET /api/match/status/{id}
│   │   │   └── results.py      # GET /api/results[/matching|/candidates]
│   │   └── services/
│   │       ├── ml_service.py   # Real ML adapter (wraps src/)
│   │       ├── file_service.py # TSV validation & storage
│   │       └── matching_service.py  # Async job management
│   ├── uploads/                # Uploaded TSV files (runtime)
│   ├── outputs/                # Generated TSV results (runtime)
│   └── requirements.txt
│
├── models/improved/            # Trained ML artifacts
│   ├── entity_matcher.pkl      # HistGradientBoosting
│   ├── name_vectorizer.pkl
│   ├── address_vectorizer.pkl
│   ├── comb_vectorizer.pkl
│   └── model_config.json       # threshold=0.70, features list
│
└── src/                        # Existing ML pipeline (unchanged)
    ├── preprocessing.py
    ├── blocking.py
    └── features.py
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/health` | Backend + ML model status |
| POST | `/api/upload` | Upload source1, source2, source3 TSV files |
| POST | `/api/match` | Start entity resolution (returns job_id) |
| GET | `/api/match/status/{job_id}` | Poll job progress |
| GET | `/api/results` | Full results with real ML confidence |
| GET | `/api/results/matching` | Download matching_results.tsv |
| GET | `/api/results/candidates` | Download candidate_pairs.tsv |

## Dataset Format

Each TSV file must contain exactly these columns:

| Column | Description |
|--------|-------------|
| `entity_id` | Unique ID (e.g. S1-00001, S2-00047) |
| `business_name` | Business name (may have typos/abbreviations) |
| `business_address` | Street address |
| `country` | Country name or ISO code |

## ML Model Integration

- **Model**: `models/improved/entity_matcher.pkl` (HistGradientBoosting)
- **Vectorizers**: `name_vectorizer.pkl`, `address_vectorizer.pkl`, `comb_vectorizer.pkl`
- **Threshold**: 0.70 (from `model_config.json`)
- **Features**: 37 signals (fuzzy ratios, TF-IDF cosines, token overlap, country flags, missing indicators)
- **Blocking**: 6-strategy multi-index (99.75% recall)

The backend **never retrains** the model. It only calls `model.predict_proba()`.

## Environment Variables

**frontend/.env**
```
VITE_API_URL=http://localhost:8000
VITE_DEMO_MODE=false
```

**backend/.env.example**
```
APP_NAME=EntitySync AI
ML_MODEL_PATH=../models/improved
UPLOAD_DIR=uploads
OUTPUT_DIR=outputs
```

## Output Files

**`matching_results.tsv`**
```
source1_entity_id    matched_entity_ids
S1-00001             S2-00001,S3-00001
S1-00004             (empty — no match)
```

**`candidate_pairs.tsv`**
```
source1_entity_id    candidate_entity_ids
S1-00001             S2-00001,S3-00001
```

Every Source 1 entity has exactly one row in both files.

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `ML model not_ready` | Verify `models/improved/` contains all 5 artifacts |
| `422 Upload Error` | Check file is `.tsv` with required columns |
| CORS error in browser | Ensure backend runs on port 8000 and `VITE_API_URL` matches |
| `ModuleNotFoundError: preprocessing` | Run backend from `backend/` dir with `PYTHONPATH` pointing to project root |
| Large dataset timeout | Blocking + feature extraction is CPU-intensive; progress is shown per stage |
