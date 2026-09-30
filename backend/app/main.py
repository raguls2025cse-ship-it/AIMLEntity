"""
main.py
FastAPI application entry point for EntitySync AI Backend.
"""

import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import CORS_ORIGINS, APP_NAME, UPLOAD_DIR, OUTPUT_DIR
from app.routes import health, upload, matching, results

# Ensure required directories exist
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Create FastAPI app
app = FastAPI(
    title=APP_NAME,
    description="Intelligent Business Entity Resolution System — AI/ML-powered deduplication across multiple data sources.",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
)

# CORS — allow local React dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount route groups under /api prefix
app.include_router(health.router, prefix="/api", tags=["Health"])
app.include_router(upload.router, prefix="/api", tags=["Upload"])
app.include_router(matching.router, prefix="/api", tags=["Matching"])
app.include_router(results.router, prefix="/api", tags=["Results"])


@app.get("/")
async def root():
    return {
        "service": APP_NAME,
        "description": "Intelligent Business Entity Resolution System",
        "docs": "/api/docs",
    }
