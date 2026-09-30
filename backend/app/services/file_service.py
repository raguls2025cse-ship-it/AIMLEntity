"""
file_service.py
Handles TSV file validation and storage.
"""

import os
import shutil
import pandas as pd
from typing import Tuple, Dict, Any, Optional
from fastapi import UploadFile

REQUIRED_COLUMNS = {"entity_id", "business_name", "business_address", "country"}
VALID_EXTENSION = ".tsv"

SOURCE_PREFIXES = {
    "source1": "S1-",
    "source2": "S2-",
    "source3": "S3-",
}


def validate_and_save_tsv(
    upload_file: UploadFile,
    save_path: str,
    source_key: str,
) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
    """
    Validates an uploaded TSV file and saves it to disk.
    Returns (success, error_message, statistics).
    """
    # 1. Check extension
    filename = upload_file.filename or ""
    _, ext = os.path.splitext(filename)
    if ext.lower() != VALID_EXTENSION:
        return False, f"File '{filename}' must have a .tsv extension.", None

    # 2. Save to a temp location for reading
    tmp_path = save_path + ".tmp"
    try:
        with open(tmp_path, "wb") as f:
            shutil.copyfileobj(upload_file.file, f)
    except Exception as e:
        return False, f"Failed to save file: {str(e)}", None

    # 3. Validate file can be read as TSV
    try:
        df = pd.read_csv(tmp_path, sep="\t", dtype=str, nrows=10000)
    except Exception as e:
        os.remove(tmp_path)
        return False, f"Failed to parse TSV file: {str(e)}", None

    # 4. Check required columns
    missing_cols = REQUIRED_COLUMNS - set(df.columns)
    if missing_cols:
        os.remove(tmp_path)
        return False, f"Missing required columns: {sorted(missing_cols)}", None

    # 5. Check entity_id column is not empty
    if df["entity_id"].isna().all() or (df["entity_id"].fillna("").str.strip() == "").all():
        os.remove(tmp_path)
        return False, "The 'entity_id' column appears to be empty.", None

    # 6. Check entity_id prefix if applicable
    expected_prefix = SOURCE_PREFIXES.get(source_key, "")
    if expected_prefix:
        sample_ids = df["entity_id"].dropna().head(5).tolist()
        if sample_ids:
            # Only warn, don't fail — the user might have different ID formats
            has_prefix = any(str(eid).startswith(expected_prefix) for eid in sample_ids)
            if not has_prefix:
                # Try to check across more rows
                df_check = pd.read_csv(tmp_path, sep="\t", dtype=str, nrows=100)
                all_ids = df_check["entity_id"].dropna().tolist()
                has_prefix = any(str(eid).startswith(expected_prefix) for eid in all_ids)
                # Not a hard failure — just a warning in the response

    # 7. Get row count (fast binary line count)
    try:
        line_count = 0
        with open(tmp_path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                line_count += chunk.count(b"\n")
        total_rows = max(0, line_count - 1)
    except Exception:
        total_rows = len(df)

    # 8. Move to final location
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    shutil.move(tmp_path, save_path)

    stats = {
        "filename": filename,
        "total_records": total_rows,
        "columns": list(df.columns),
    }

    return True, "OK", stats
