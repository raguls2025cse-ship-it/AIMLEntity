"""
matching_service.py
Async job management for entity resolution pipeline.
"""

import asyncio
import uuid
from typing import Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor

from app.services.ml_service import run_entity_resolution, get_job_status
from app.config import SOURCE1_PATH, SOURCE2_PATH, SOURCE3_PATH

_executor = ThreadPoolExecutor(max_workers=2)


def create_job_id() -> str:
    return str(uuid.uuid4())[:8]


def start_matching_job() -> str:
    """
    Starts entity resolution in a background thread.
    Returns a job_id that can be polled via /api/match/status/{job_id}.
    """
    job_id = create_job_id()

    def _run():
        try:
            run_entity_resolution(
                source1_path=SOURCE1_PATH,
                source2_path=SOURCE2_PATH,
                source3_path=SOURCE3_PATH,
                job_id=job_id,
            )
        except Exception as e:
            from app.services.ml_service import _update_job
            _update_job(job_id, "Failed", 0, status="failed", error=str(e))

    _executor.submit(_run)
    return job_id


def get_matching_status(job_id: str) -> Optional[Dict[str, Any]]:
    return get_job_status(job_id)
