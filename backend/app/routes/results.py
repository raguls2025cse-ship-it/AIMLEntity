"""
results.py
Results API endpoints — statistics, data, and file downloads.
"""

import os
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse, FileResponse

from app.services.ml_service import get_results_with_details
from app.config import MATCHING_RESULTS_PATH, CANDIDATE_PAIRS_PATH

router = APIRouter()


@router.get("/results")
async def get_results():
    """
    Returns processed matching statistics and detailed result rows.
    Includes real confidence values from the ML model.
    """
    data = get_results_with_details()
    if not data.get("available"):
        raise HTTPException(
            status_code=404,
            detail="No matching results found. Please run the entity resolution pipeline first.",
        )
    return JSONResponse(status_code=200, content=data)


@router.get("/results/matching")
async def download_matching_results():
    """
    Download matching_results.tsv.
    """
    if not os.path.exists(MATCHING_RESULTS_PATH):
        raise HTTPException(
            status_code=404,
            detail="matching_results.tsv not found. Run entity resolution first.",
        )
    return FileResponse(
        path=MATCHING_RESULTS_PATH,
        media_type="text/tab-separated-values",
        filename="matching_results.tsv",
    )


@router.get("/results/candidates")
async def download_candidate_pairs():
    """
    Download candidate_pairs.tsv.
    """
    if not os.path.exists(CANDIDATE_PAIRS_PATH):
        raise HTTPException(
            status_code=404,
            detail="candidate_pairs.tsv not found. Run entity resolution first.",
        )
    return FileResponse(
        path=CANDIDATE_PAIRS_PATH,
        media_type="text/tab-separated-values",
        filename="candidate_pairs.tsv",
    )
