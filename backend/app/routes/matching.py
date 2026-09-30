"""
matching.py
Entity matching API endpoints.
"""

import os
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

from app.services.matching_service import start_matching_job, get_matching_status
from app.services.ml_service import get_model_status
from app.config import SOURCE1_PATH, SOURCE2_PATH, SOURCE3_PATH

router = APIRouter()


@router.post("/match")
async def start_matching():
    """
    Starts the entity resolution pipeline asynchronously.
    Returns a job_id that can be polled for status.
    """
    # Verify uploaded files exist
    missing_files = []
    for name, path in [
        ("source1", SOURCE1_PATH),
        ("source2", SOURCE2_PATH),
        ("source3", SOURCE3_PATH),
    ]:
        if not os.path.exists(path):
            missing_files.append(name)

    if missing_files:
        raise HTTPException(
            status_code=400,
            detail=f"Missing uploaded source files: {missing_files}. Please upload all three sources first.",
        )

    # Verify ML artifacts
    model_status = get_model_status()
    if not model_status["ready"]:
        raise HTTPException(
            status_code=503,
            detail=f"ML model is not ready: {model_status['message']}",
        )

    # Start background job
    job_id = start_matching_job()

    return JSONResponse(
        status_code=202,
        content={
            "success": True,
            "status": "started",
            "job_id": job_id,
            "message": "Entity resolution pipeline started.",
        },
    )


@router.get("/match/status/{job_id}")
async def check_matching_status(job_id: str):
    """
    Poll the status of a running or completed matching job.
    """
    status = get_matching_status(job_id)
    if status is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")

    return JSONResponse(status_code=200, content=status)
