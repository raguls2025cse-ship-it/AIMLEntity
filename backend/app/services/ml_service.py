"""
ml_service.py
Real ML adapter service for EntitySync AI.
Wraps the existing preprocessing, blocking, features, and trained model.
"""

import sys
import os
import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple
from pathlib import Path

# --- Inject the existing src/ directory into Python path so we can import
#     preprocessing.py, blocking.py, features.py exactly as they were used
#     during training (no duplication).
from app.config import (
    SRC_DIR, MODEL_PATH, NAME_VEC_PATH, ADDR_VEC_PATH,
    COMB_VEC_PATH, CONFIG_PATH,
    MATCHING_RESULTS_PATH, CANDIDATE_PAIRS_PATH, OUTPUT_DIR
)

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# These are the existing ML pipeline modules — imported exactly as-is
from preprocessing import preprocess_dataframe          # noqa: E402
from blocking import generate_candidate_pairs            # noqa: E402
from features import extract_features_for_pairs          # noqa: E402

# ---------------------------------------------------------------------------
# Module-level model cache — loaded once per process
# ---------------------------------------------------------------------------
_model = None
_name_vec = None
_addr_vec = None
_comb_vec = None
_model_config = None
_model_ready = False


def initialize_model() -> Tuple[bool, str]:
    """
    Verifies and loads all ML artifacts into memory exactly once.
    Returns (success: bool, message: str).
    """
    global _model, _name_vec, _addr_vec, _comb_vec, _model_config, _model_ready

    if _model_ready:
        return True, "Model already loaded."

    missing = []
    for label, path in [
        ("entity_matcher.pkl", MODEL_PATH),
        ("name_vectorizer.pkl", NAME_VEC_PATH),
        ("address_vectorizer.pkl", ADDR_VEC_PATH),
        ("comb_vectorizer.pkl", COMB_VEC_PATH),
        ("model_config.json", CONFIG_PATH),
    ]:
        if not os.path.exists(path):
            missing.append(f"{label} (expected: {path})")

    if missing:
        msg = "ML model configuration is incomplete. Missing: " + ", ".join(missing)
        return False, msg

    try:
        with open(CONFIG_PATH, "r") as f:
            _model_config = json.load(f)

        _model = joblib.load(MODEL_PATH)
        _name_vec = joblib.load(NAME_VEC_PATH)
        _addr_vec = joblib.load(ADDR_VEC_PATH)
        _comb_vec = joblib.load(COMB_VEC_PATH)

        _model_ready = True
        return True, "ML model loaded successfully."
    except Exception as e:
        return False, f"Failed to load ML artifacts: {str(e)}"


def get_model_status() -> Dict[str, Any]:
    """Return a dict describing model readiness."""
    ok, msg = initialize_model()
    return {
        "ready": ok,
        "message": msg,
        "model_name": _model_config.get("model_name", "unknown") if _model_config else None,
        "threshold": _model_config.get("best_threshold", 0.70) if _model_config else None,
        "feature_count": len(_model_config.get("features", [])) if _model_config else None,
    }


# ---------------------------------------------------------------------------
# Job status tracking (in-memory for single-process server)
# ---------------------------------------------------------------------------
_job_status: Dict[str, Dict[str, Any]] = {}


def get_job_status(job_id: str) -> Optional[Dict[str, Any]]:
    return _job_status.get(job_id)


def _update_job(job_id: str, stage: str, progress: int, status: str = "running",
                statistics: Optional[Dict] = None, error: Optional[str] = None):
    _job_status[job_id] = {
        "job_id": job_id,
        "status": status,
        "stage": stage,
        "progress": progress,
        "statistics": statistics or {},
        "error": error,
    }


# ---------------------------------------------------------------------------
# Main inference pipeline
# ---------------------------------------------------------------------------

def run_entity_resolution(
    source1_path: str,
    source2_path: str,
    source3_path: str,
    job_id: str = "default",
) -> Dict[str, Any]:
    """
    Full entity resolution pipeline:
      1. Load + validate TSV files
      2. Preprocess (reusing existing preprocessing.py)
      3. Candidate generation / blocking (reusing existing blocking.py)
      4. Feature extraction (reusing existing features.py)
      5. Trained model inference
      6. Threshold application
      7. Output file generation
    Returns statistics dict.
    """

    # --- 0. Ensure model is loaded ---
    _update_job(job_id, "Initializing ML model", 5)
    ok, msg = initialize_model()
    if not ok:
        _update_job(job_id, "Failed", 0, status="failed", error=msg)
        raise RuntimeError(msg)

    threshold = float(_model_config.get("best_threshold", 0.70))
    feature_names = _model_config.get("features", [])

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- 1. Load source TSVs (engine='c' = fastest pandas read) ---
    _update_job(job_id, "Loading source datasets", 10)
    df_s1 = pd.read_csv(source1_path, sep="\t", dtype=str, engine="c", low_memory=False).fillna("")
    df_s2 = pd.read_csv(source2_path, sep="\t", dtype=str, engine="c", low_memory=False).fillna("")
    df_s3 = pd.read_csv(source3_path, sep="\t", dtype=str, engine="c", low_memory=False).fillna("")

    print(f"[EntityResolution] Loaded: S1={len(df_s1):,}, S2={len(df_s2):,}, S3={len(df_s3):,}")

    # --- 2. Preprocess (existing preprocessing.py) ---
    _update_job(job_id, "Preprocessing records", 20)
    df_s1_proc = preprocess_dataframe(df_s1)
    df_s2_proc = preprocess_dataframe(df_s2)
    df_s3_proc = preprocess_dataframe(df_s3)

    # --- 3. Candidate generation / blocking (existing blocking.py) ---
    _update_job(job_id, "Generating candidate pairs (blocking)", 35)
    df_pairs = generate_candidate_pairs(df_s1_proc, df_s2_proc, df_s3_proc)

    total_candidates = len(df_pairs)
    print(f"[EntityResolution] Candidate pairs generated: {total_candidates:,}")

    # --- 4. Build entity dicts (vectorized — much faster than iterrows on large DFs) ---
    _update_job(job_id, "Extracting features", 55)
    s1_dict = df_s1_proc.set_index("entity_id").to_dict(orient="index")
    cand_dict = pd.concat([df_s2_proc, df_s3_proc], ignore_index=True).set_index("entity_id").to_dict(orient="index")

    # --- 5. Feature extraction (existing features.py, large batch_size = fewer matrix ops) ---
    _update_job(job_id, "Extracting features", 60)
    df_p_info, X_feat, _, f_names_out = extract_features_for_pairs(
        df_pairs=df_pairs,
        s1_dict=s1_dict,
        cand_dict=cand_dict,
        name_vec=_name_vec,
        addr_vec=_addr_vec,
        comb_vec=_comb_vec,
        batch_size=500000,  # larger batches = faster sparse matrix multiply
    )

    # --- 6. Reorder features to match model training order (from model_config.json) ---
    if feature_names and f_names_out:
        feat_idx = {name: i for i, name in enumerate(f_names_out)}
        col_order = [feat_idx[fn] for fn in feature_names if fn in feat_idx]
        if len(col_order) == len(feature_names):
            X_feat = X_feat[:, col_order]

    # --- 7. Model inference ---
    _update_job(job_id, "Running ML model inference", 75)
    probs = _model.predict_proba(X_feat)[:, 1]

    # --- 8. Apply threshold ---
    _update_job(job_id, "Applying threshold and generating results", 85)
    predictions = (probs >= threshold).astype(int)

    df_pairs = df_pairs.copy()
    df_pairs["probability"] = probs
    df_pairs["predicted_match"] = predictions

    # Matched pairs only
    df_matched = df_pairs[df_pairs["predicted_match"] == 1].copy()

    # --- 8b. Save pairs with probabilities for later confidence lookup ---
    pairs_with_probs_path = os.path.join(OUTPUT_DIR, "pairs_with_probs.tsv")
    df_pairs[["source1_entity_id", "candidate_entity_id", "probability", "predicted_match"]].to_csv(
        pairs_with_probs_path, sep="\t", index=False
    )

    # --- 9. Generate candidate_pairs.tsv ---
    # Format: source1_entity_id \t candidate_entity_ids (comma-separated)
    _update_job(job_id, "Generating output files", 90)

    s1_all_ids = df_s1_proc["entity_id"].tolist()

    # candidate_pairs: all S1 ids, even those with no candidates
    cand_grouped = (
        df_pairs.groupby("source1_entity_id")["candidate_entity_id"]
        .apply(lambda x: ",".join(sorted(set(x.tolist()))))
        .reset_index()
        .rename(columns={"candidate_entity_id": "candidate_entity_ids"})
    )
    cand_df_full = pd.DataFrame({"source1_entity_id": s1_all_ids})
    cand_df_full = cand_df_full.merge(cand_grouped, on="source1_entity_id", how="left")
    cand_df_full["candidate_entity_ids"] = cand_df_full["candidate_entity_ids"].fillna("")
    cand_df_full.to_csv(CANDIDATE_PAIRS_PATH, sep="\t", index=False)

    # matching_results: S1 ids with their matched entity IDs
    match_grouped = (
        df_matched.groupby("source1_entity_id")["candidate_entity_id"]
        .apply(lambda x: ",".join(sorted(set(x.tolist()))))
        .reset_index()
        .rename(columns={"candidate_entity_id": "matched_entity_ids"})
    )
    match_df_full = pd.DataFrame({"source1_entity_id": s1_all_ids})
    match_df_full = match_df_full.merge(match_grouped, on="source1_entity_id", how="left")
    match_df_full["matched_entity_ids"] = match_df_full["matched_entity_ids"].fillna("")
    match_df_full.to_csv(MATCHING_RESULTS_PATH, sep="\t", index=False)

    # --- 10. Compute statistics ---
    matched_s1_ids = set(df_matched["source1_entity_id"].unique())
    matched_count = len(matched_s1_ids)
    total_s1 = len(s1_all_ids)
    unmatched_count = total_s1 - matched_count
    match_rate = (matched_count / total_s1 * 100) if total_s1 > 0 else 0.0

    stats = {
        "source1_entities": total_s1,
        "source2_records": len(df_s2),
        "source3_records": len(df_s3),
        "matched_entities": matched_count,
        "unmatched_entities": unmatched_count,
        "match_rate": round(match_rate, 2),
        "candidate_pairs": total_candidates,
        "threshold_used": threshold,
    }

    print(f"[EntityResolution] Done. Matched S1 entities: {matched_count}/{total_s1}")
    _update_job(job_id, "Completed", 100, status="completed", statistics=stats)
    return stats


# ---------------------------------------------------------------------------
# Results reader (used by /api/results endpoint)
# ---------------------------------------------------------------------------

def get_results_with_details() -> Dict[str, Any]:
    """
    Reads matching_results.tsv and returns enriched result rows,
    joining with uploaded source files for business name etc.
    """
    from app.config import SOURCE1_PATH, SOURCE2_PATH, SOURCE3_PATH

    if not os.path.exists(MATCHING_RESULTS_PATH):
        return {"available": False, "rows": [], "statistics": {}}

    df_results = pd.read_csv(MATCHING_RESULTS_PATH, sep="\t", dtype=str).fillna("")

    # Load source data for enrichment
    def safe_read(path):
        if os.path.exists(path):
            return pd.read_csv(path, sep="\t", dtype=str).fillna("")
        return pd.DataFrame()

    df_s1 = safe_read(SOURCE1_PATH)
    df_s2 = safe_read(SOURCE2_PATH)
    df_s3 = safe_read(SOURCE3_PATH)

    s1_lookup = {}
    if not df_s1.empty:
        for _, row in df_s1.iterrows():
            s1_lookup[row.get("entity_id", "")] = {
                "business_name": row.get("business_name", ""),
                "business_address": row.get("business_address", ""),
                "country": row.get("country", ""),
            }

    cand_lookup = {}
    for df_src in [df_s2, df_s3]:
        if not df_src.empty:
            for _, row in df_src.iterrows():
                cand_lookup[row.get("entity_id", "")] = {
                    "business_name": row.get("business_name", ""),
                    "business_address": row.get("business_address", ""),
                    "country": row.get("country", ""),
                }

    # Load candidate pairs for confidence
    df_candidates = pd.DataFrame()
    if os.path.exists(CANDIDATE_PAIRS_PATH):
        try:
            df_candidates = pd.read_csv(CANDIDATE_PAIRS_PATH, sep="\t", dtype=str).fillna("")
        except Exception:
            pass

    # Load probabilities from pairs if available (stored in matching run)
    prob_lookup = {}
    pairs_with_probs_path = os.path.join(OUTPUT_DIR, "pairs_with_probs.tsv")
    if os.path.exists(pairs_with_probs_path):
        try:
            df_probs = pd.read_csv(pairs_with_probs_path, sep="\t", dtype=str).fillna("")
            for _, row in df_probs.iterrows():
                key = (row.get("source1_entity_id", ""), row.get("candidate_entity_id", ""))
                try:
                    prob_lookup[key] = float(row.get("probability", 0))
                except (ValueError, TypeError):
                    prob_lookup[key] = None
        except Exception:
            pass

    rows = []
    matched_count = 0
    for _, r in df_results.iterrows():
        s1_id = r.get("source1_entity_id", "")
        matched_ids_str = r.get("matched_entity_ids", "")
        matched_ids = [m.strip() for m in matched_ids_str.split(",") if m.strip()] if matched_ids_str else []

        s1_info = s1_lookup.get(s1_id, {})
        is_matched = len(matched_ids) > 0
        if is_matched:
            matched_count += 1

        matched_details = []
        for mid in matched_ids:
            m_info = cand_lookup.get(mid, {})
            prob = prob_lookup.get((s1_id, mid))
            matched_details.append({
                "entity_id": mid,
                "business_name": m_info.get("business_name", ""),
                "business_address": m_info.get("business_address", ""),
                "country": m_info.get("country", ""),
                "confidence": round(prob, 4) if prob is not None else None,
                "source": "S2" if mid.startswith("S2-") else ("S3" if mid.startswith("S3-") else "unknown"),
            })

        rows.append({
            "source1_entity_id": s1_id,
            "business_name": s1_info.get("business_name", ""),
            "business_address": s1_info.get("business_address", ""),
            "country": s1_info.get("country", ""),
            "matched_entity_ids": matched_ids_str,
            "matched_details": matched_details,
            "status": "matched" if is_matched else "unmatched",
        })

    total = len(rows)
    unmatched = total - matched_count
    match_rate = round(matched_count / total * 100, 2) if total > 0 else 0.0

    statistics = {
        "source1_entities": total,
        "matched_entities": matched_count,
        "unmatched_entities": unmatched,
        "match_rate": match_rate,
    }

    return {"available": True, "rows": rows, "statistics": statistics}
