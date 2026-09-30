"""
preprocessing.py
Safe, standardized text and entity normalization for:
- business_name
- business_address
- country
Preserves both raw and normalized values without over-normalizing.
"""

import re
import unicodedata
from typing import List, Dict, Any, Optional
import pandas as pd


# Common business suffix abbreviations mapping to normalized form
BUSINESS_SUFFIX_MAP = {
    "incorporated": "inc",
    "corporation": "corp",
    "limited": "ltd",
    "company": "co",
    "private limited": "pvt ltd",
    "private": "pvt",
    "enterprises": "ent",
    "enterprise": "ent",
    "services": "svc",
    "service": "svc",
    "technologies": "tech",
    "technology": "tech",
    "international": "intl",
    "associates": "assoc",
    "consulting": "consult",
    "group": "grp",
}

# Common address abbreviations mapping
ADDRESS_ABBR_MAP = {
    "street": "st",
    "str": "st",
    "avenue": "ave",
    "av": "ave",
    "road": "rd",
    "boulevard": "blvd",
    "drive": "dr",
    "lane": "ln",
    "court": "ct",
    "place": "pl",
    "parkway": "pkwy",
    "highway": "hwy",
    "suite": "ste",
    "apartment": "apt",
    "floor": "fl",
    "building": "bldg",
    "post office box": "pobox",
    "p o box": "pobox",
    "po box": "pobox",
    "pobox": "pobox",
    "north": "n",
    "south": "s",
    "east": "e",
    "west": "w",
    "northeast": "ne",
    "northwest": "nw",
    "southeast": "se",
    "southwest": "sw",
}

# Country normalization map
COUNTRY_SYNONYMS = {
    "us": "united states",
    "usa": "united states",
    "u.s.": "united states",
    "u.s.a.": "united states",
    "united states": "united states",
    "united states of america": "united states",
    "in": "india",
    "ind": "india",
    "india": "india",
    "uk": "united kingdom",
    "gb": "united kingdom",
    "gbr": "united kingdom",
    "great britain": "united kingdom",
    "united kingdom": "united kingdom",
    "ca": "canada",
    "can": "canada",
    "canada": "canada",
    "au": "australia",
    "aus": "australia",
    "australia": "australia",
    "de": "germany",
    "deu": "germany",
    "germany": "germany",
    "fr": "france",
    "fra": "france",
    "france": "france",
    "sg": "singapore",
    "sgp": "singapore",
    "singapore": "singapore",
    "ae": "united arab emirates",
    "uae": "united arab emirates",
    "united arab emirates": "united arab emirates",
}


def safe_unicode_normalize(text: Any) -> str:
    """Safely normalizes Unicode text to ASCII compatibility without data loss."""
    if text is None or pd.isna(text):
        return ""
    text_str = str(text)
    # NFKD decomposition
    normalized = unicodedata.normalize("NFKD", text_str)
    # Convert non-ascii equivalents safely
    ascii_text = normalized.encode("ascii", "ignore").decode("utf-8")
    return ascii_text if ascii_text.strip() else text_str


def normalize_business_name(raw_name: Any) -> str:
    """
    Normalizes business name:
    - Lowercase
    - Safe Unicode cleanup
    - Replace punctuation with spaces
    - Standardize common business suffixes
    - Remove extra whitespace
    """
    text = safe_unicode_normalize(raw_name).lower().strip()
    if not text:
        return ""

    # Replace special characters & punctuation with space, preserving alphanumeric
    text = re.sub(r"[^\w\s]", " ", text)
    tokens = text.split()

    normalized_tokens = []
    for token in tokens:
        clean_token = token.strip()
        if not clean_token:
            continue
        mapped = BUSINESS_SUFFIX_MAP.get(clean_token, clean_token)
        normalized_tokens.append(mapped)

    return " ".join(normalized_tokens)


def normalize_address(raw_address: Any) -> str:
    """
    Normalizes business address:
    - Lowercase
    - Safe Unicode cleanup
    - Standardize street/suite/directional abbreviations
    - Clean punctuation and extra spaces
    """
    text = safe_unicode_normalize(raw_address).lower().strip()
    if not text:
        return ""

    # Normalize PO Box variations first
    text = re.sub(r"\bp\s*\.?\s*o\s*\.?\s*box\b", "pobox", text)
    # Replace special punctuation with space
    text = re.sub(r"[^\w\s]", " ", text)
    tokens = text.split()

    normalized_tokens = []
    for token in tokens:
        clean_token = token.strip()
        if not clean_token:
            continue
        mapped = ADDRESS_ABBR_MAP.get(clean_token, clean_token)
        normalized_tokens.append(mapped)

    return " ".join(normalized_tokens)


def normalize_country(raw_country: Any) -> str:
    """Normalizes country string to a canonical representation."""
    if raw_country is None or pd.isna(raw_country):
        return ""
    clean = safe_unicode_normalize(raw_country).lower().strip()
    clean = re.sub(r"[^\w\s]", "", clean).strip()
    return COUNTRY_SYNONYMS.get(clean, clean)


def tokenize_text(text: str) -> List[str]:
    """Extracts non-empty alphanumeric tokens."""
    if not text:
        return []
    return [t for t in text.split() if t]


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocesses business entity records.
    Adds normalized fields while preserving raw fields.
    """
    df_out = df.copy()
    
    # Ensure standard string columns
    for col in ["business_name", "business_address", "country"]:
        if col not in df_out.columns:
            df_out[col] = ""
        else:
            df_out[col] = df_out[col].fillna("").astype(str)

    # Add normalized columns
    df_out["name_norm"] = df_out["business_name"].apply(normalize_business_name)
    df_out["address_norm"] = df_out["business_address"].apply(normalize_address)
    df_out["country_norm"] = df_out["country"].apply(normalize_country)
    
    # Token columns for fast set operations
    df_out["name_tokens"] = df_out["name_norm"].apply(tokenize_text)
    df_out["addr_tokens"] = df_out["address_norm"].apply(tokenize_text)

    return df_out
