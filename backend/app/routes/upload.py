"""
upload.py
File upload API endpoint.
"""

import os
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse

from app.services.file_service import validate_and_save_tsv
from app.config import SOURCE1_PATH, SOURCE2_PATH, SOURCE3_PATH

router = APIRouter()


@router.post("/upload")
async def upload_datasets(
    source1: UploadFile = File(...),
    source2: UploadFile = File(...),
    source3: UploadFile = File(...),
):
    """
    Upload all three TSV source files for entity resolution.
    Each file must have columns: entity_id, business_name, business_address, country.
    """
    results = {}
    errors = []

    for key, upload_file, save_path in [
        ("source1", source1, SOURCE1_PATH),
        ("source2", source2, SOURCE2_PATH),
        ("source3", source3, SOURCE3_PATH),
    ]:
        success, msg, stats = validate_and_save_tsv(upload_file, save_path, key)
        if not success:
            errors.append(f"{key}: {msg}")
        else:
            results[key] = stats

    if errors:
        raise HTTPException(
            status_code=422,
            detail={"message": "Upload validation failed.", "errors": errors},
        )

    return JSONResponse(
        status_code=200,
        content={
            "success": True,
            "message": "Datasets uploaded successfully",
            "statistics": {
                "source1_records": results["source1"]["total_records"],
                "source2_records": results["source2"]["total_records"],
                "source3_records": results["source3"]["total_records"],
            },
            "files": results,
        },
    )
