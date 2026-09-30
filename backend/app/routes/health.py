"""
health.py
Health check API endpoint.
"""

from fastapi import APIRouter
from app.services.ml_service import initialize_model, get_model_status

router = APIRouter()


@router.get("/health")
async def health_check():
    """
    Returns service health status including ML model readiness.
    The ml_model status is determined by actually attempting to load the artifacts.
    """
    model_status = get_model_status()
    return {
        "status": "ok",
        "service": "EntitySync AI Backend",
        "ml_model": "ready" if model_status["ready"] else "not_ready",
        "ml_model_details": {
            "message": model_status["message"],
            "model_name": model_status.get("model_name"),
            "threshold": model_status.get("threshold"),
            "feature_count": model_status.get("feature_count"),
        },
    }
