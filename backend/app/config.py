"""
config.py
Centralized configuration for EntitySync AI Backend.
"""

import os
from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent.parent  # EntitySync root
BACKEND_DIR = Path(__file__).resolve().parent.parent      # backend/

# ML Model paths — use the improved/ subdirectory
ML_MODEL_DIR = os.environ.get("ML_MODEL_PATH", str(BASE_DIR / "models" / "improved"))
MODEL_PATH = os.path.join(ML_MODEL_DIR, "entity_matcher.pkl")
NAME_VEC_PATH = os.path.join(ML_MODEL_DIR, "name_vectorizer.pkl")
ADDR_VEC_PATH = os.path.join(ML_MODEL_DIR, "address_vectorizer.pkl")
COMB_VEC_PATH = os.path.join(ML_MODEL_DIR, "comb_vectorizer.pkl")
CONFIG_PATH = os.path.join(ML_MODEL_DIR, "model_config.json")

# Source ML modules directory (existing pipeline)
SRC_DIR = str(BASE_DIR / "src")

# Upload / output directories
UPLOAD_DIR = os.environ.get("UPLOAD_DIR", str(BACKEND_DIR / "uploads"))
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", str(BACKEND_DIR / "outputs"))

# Matching output files
MATCHING_RESULTS_PATH = os.path.join(OUTPUT_DIR, "matching_results.tsv")
CANDIDATE_PAIRS_PATH = os.path.join(OUTPUT_DIR, "candidate_pairs.tsv")

# Uploaded file paths (per session – overwritten each run)
SOURCE1_PATH = os.path.join(UPLOAD_DIR, "source1.tsv")
SOURCE2_PATH = os.path.join(UPLOAD_DIR, "source2.tsv")
SOURCE3_PATH = os.path.join(UPLOAD_DIR, "source3.tsv")

# CORS origins
CORS_ORIGINS = [
    "*"
]

APP_NAME = os.environ.get("APP_NAME", "EntitySync AI")
