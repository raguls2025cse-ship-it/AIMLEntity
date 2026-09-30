"""
features.py
Comprehensive feature engineering module for Entity Matching.
Computes 32 rich tabular similarity and compatibility signals:
- Business name metrics (RapidFuzz ratio, WRatio, QRatio, token sort/set, partial ratio, char n-gram TF-IDF, token overlap, prefix match)
- Address metrics (RapidFuzz ratio, WRatio, QRatio, token sort/set, partial ratio, char n-gram TF-IDF, numeric token match)
- Cross-field & Country metrics (country exact/normalized/conflict, combined TF-IDF cosine, weighted similarity, shared token counts)
- Missing value indicators (explicit binary flags for absent name/address/country)
"""

import re
from typing import Dict, List, Set, Tuple, Any, Optional
import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer


def jaccard_similarity(tokens1: List[str], tokens2: List[str]) -> Tuple[float, int]:
    """Computes Jaccard similarity and intersection count between two token lists."""
    s1 = set(tokens1)
    s2 = set(tokens2)
    if not s1 and not s2:
        return 1.0, 0
    if not s1 or not s2:
        return 0.0, 0
    intersection = len(s1.intersection(s2))
    union = len(s1.union(s2))
    return (intersection / union if union > 0 else 0.0), intersection


def numeric_overlap_similarity(text1: str, text2: str) -> float:
    """Checks whether numbers (street/suite numbers) in text1 and text2 match."""
    nums1 = set(re.findall(r"\b\d+\b", text1))
    nums2 = set(re.findall(r"\b\d+\b", text2))
    if not nums1 or not nums2:
        return 0.5  # Neutral if one/both lack numbers
    if nums1.intersection(nums2):
        return 1.0  # Shared number
    return 0.0  # Conflicting numbers


def build_vectorizers(
    all_names: List[str],
    all_addresses: List[str],
    all_combined: Optional[List[str]] = None,
) -> Tuple[TfidfVectorizer, TfidfVectorizer, TfidfVectorizer]:
    """Fits TF-IDF vectorizers on names, addresses, and combined strings."""
    name_vec = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 4),
        min_df=1,
        sublinear_tf=True
    )
    name_vec.fit(all_names)

    addr_vec = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 4),
        min_df=1,
        sublinear_tf=True
    )
    addr_vec.fit(all_addresses)

    if all_combined is None:
        all_combined = [f"{n} {a}".strip() for n, a in zip(all_names, all_addresses)]

    comb_vec = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 4),
        min_df=1,
        sublinear_tf=True
    )
    comb_vec.fit(all_combined)

    return name_vec, addr_vec, comb_vec


def extract_features_for_pairs(
    df_pairs: pd.DataFrame,
    s1_dict: Dict[str, Dict[str, Any]],
    cand_dict: Dict[str, Dict[str, Any]],
    name_vec: Optional[TfidfVectorizer] = None,
    addr_vec: Optional[TfidfVectorizer] = None,
    comb_vec: Optional[TfidfVectorizer] = None,
    true_pairs_set: Optional[Set[Tuple[str, str]]] = None,
    batch_size: int = 50000,
) -> Tuple[pd.DataFrame, np.ndarray, Optional[np.ndarray], List[str]]:
    """
    Computes comprehensive feature matrix for candidate pairs.
    """
    feature_names = [
        # 1. Business Name Features (12)
        "name_fuzz_ratio",
        "name_fuzz_wratio",
        "name_fuzz_qratio",
        "name_token_sort_ratio",
        "name_token_set_ratio",
        "name_partial_ratio",
        "name_len_diff",
        "name_len_ratio",
        "name_token_jaccard",
        "name_common_tokens_count",
        "name_tfidf_cosine",
        "name_prefix_match",
        # 2. Address Features (11)
        "addr_fuzz_ratio",
        "addr_fuzz_wratio",
        "addr_fuzz_qratio",
        "addr_token_sort_ratio",
        "addr_token_set_ratio",
        "addr_partial_ratio",
        "addr_len_diff",
        "addr_len_ratio",
        "addr_token_jaccard",
        "addr_common_tokens_count",
        "addr_tfidf_cosine",
        "addr_numeric_overlap",
        # 3. Cross-Field & Combined Features (6)
        "comb_tfidf_cosine",
        "country_exact_match",
        "country_norm_match",
        "country_conflict",
        "combined_weighted_sim",
        "total_shared_tokens",
        "is_source_3",
        # 4. Missing Value Indicators (6)
        "s1_name_missing",
        "cand_name_missing",
        "s1_addr_missing",
        "cand_addr_missing",
        "s1_country_missing",
        "cand_country_missing",
    ]

    n_pairs = len(df_pairs)
    print(f"Extracting {len(feature_names)} features for {n_pairs:,} candidate pairs...")
    
    s1_ids = df_pairs["source1_entity_id"].values
    cand_ids = df_pairs["candidate_entity_id"].values

    features = np.zeros((n_pairs, len(feature_names)), dtype=np.float32)

    # Batch cosine similarity computation
    name_tfidf_sims = np.zeros(n_pairs, dtype=np.float32)
    addr_tfidf_sims = np.zeros(n_pairs, dtype=np.float32)
    comb_tfidf_sims = np.zeros(n_pairs, dtype=np.float32)

    s1_names_list = [s1_dict[s1_ids[i]]["name_norm"] for i in range(n_pairs)]
    cand_names_list = [cand_dict[cand_ids[i]]["name_norm"] for i in range(n_pairs)]
    s1_addrs_list = [s1_dict[s1_ids[i]]["address_norm"] for i in range(n_pairs)]
    cand_addrs_list = [cand_dict[cand_ids[i]]["address_norm"] for i in range(n_pairs)]

    if name_vec is not None:
        for b_start in range(0, n_pairs, batch_size):
            b_end = min(b_start + batch_size, n_pairs)
            s1_mat = name_vec.transform(s1_names_list[b_start:b_end])
            c_mat = name_vec.transform(cand_names_list[b_start:b_end])
            name_tfidf_sims[b_start:b_end] = np.asarray(s1_mat.multiply(c_mat).sum(axis=1)).ravel()

    if addr_vec is not None:
        for b_start in range(0, n_pairs, batch_size):
            b_end = min(b_start + batch_size, n_pairs)
            s1_mat = addr_vec.transform(s1_addrs_list[b_start:b_end])
            c_mat = addr_vec.transform(cand_addrs_list[b_start:b_end])
            addr_tfidf_sims[b_start:b_end] = np.asarray(s1_mat.multiply(c_mat).sum(axis=1)).ravel()

    if comb_vec is not None:
        s1_comb_list = [f"{n} {a}".strip() for n, a in zip(s1_names_list, s1_addrs_list)]
        cand_comb_list = [f"{n} {a}".strip() for n, a in zip(cand_names_list, cand_addrs_list)]
        for b_start in range(0, n_pairs, batch_size):
            b_end = min(b_start + batch_size, n_pairs)
            s1_mat = comb_vec.transform(s1_comb_list[b_start:b_end])
            c_mat = comb_vec.transform(cand_comb_list[b_start:b_end])
            comb_tfidf_sims[b_start:b_end] = np.asarray(s1_mat.multiply(c_mat).sum(axis=1)).ravel()

    # Fast row-wise feature calculation
    for i in range(n_pairs):
        s1 = s1_dict[s1_ids[i]]
        cand = cand_dict[cand_ids[i]]

        s1_name = s1_names_list[i]
        cand_name = cand_names_list[i]
        s1_addr = s1_addrs_list[i]
        cand_addr = cand_addrs_list[i]

        s1_country_raw = s1.get("country", "")
        cand_country_raw = cand.get("country", "")
        s1_country_norm = s1.get("country_norm", "")
        cand_country_norm = cand.get("country_norm", "")

        # 1. Name Features
        name_ratio = fuzz.ratio(s1_name, cand_name) / 100.0 if s1_name and cand_name else 0.0
        name_wratio = fuzz.WRatio(s1_name, cand_name) / 100.0 if s1_name and cand_name else 0.0
        name_qratio = fuzz.QRatio(s1_name, cand_name) / 100.0 if s1_name and cand_name else 0.0
        name_sort = fuzz.token_sort_ratio(s1_name, cand_name) / 100.0 if s1_name and cand_name else 0.0
        name_set = fuzz.token_set_ratio(s1_name, cand_name) / 100.0 if s1_name and cand_name else 0.0
        name_partial = fuzz.partial_ratio(s1_name, cand_name) / 100.0 if s1_name and cand_name else 0.0
        name_len_diff = abs(len(s1_name) - len(cand_name))
        max_name_len = max(len(s1_name), len(cand_name), 1)
        name_len_ratio = min(len(s1_name), len(cand_name)) / max_name_len
        name_jaccard, name_common_cnt = jaccard_similarity(s1["name_tokens"], cand["name_tokens"])
        name_tfidf = name_tfidf_sims[i]

        # Prefix Match
        prefix_len = 0
        min_l = min(len(s1_name), len(cand_name))
        while prefix_len < min_l and s1_name[prefix_len] == cand_name[prefix_len]:
            prefix_len += 1
        name_prefix_match = prefix_len / min_l if min_l > 0 else 0.0

        # 2. Address Features
        addr_ratio = fuzz.ratio(s1_addr, cand_addr) / 100.0 if s1_addr and cand_addr else 0.0
        addr_wratio = fuzz.WRatio(s1_addr, cand_addr) / 100.0 if s1_addr and cand_addr else 0.0
        addr_qratio = fuzz.QRatio(s1_addr, cand_addr) / 100.0 if s1_addr and cand_addr else 0.0
        addr_sort = fuzz.token_sort_ratio(s1_addr, cand_addr) / 100.0 if s1_addr and cand_addr else 0.0
        addr_set = fuzz.token_set_ratio(s1_addr, cand_addr) / 100.0 if s1_addr and cand_addr else 0.0
        addr_partial = fuzz.partial_ratio(s1_addr, cand_addr) / 100.0 if s1_addr and cand_addr else 0.0
        addr_len_diff = abs(len(s1_addr) - len(cand_addr))
        max_addr_len = max(len(s1_addr), len(cand_addr), 1)
        addr_len_ratio = min(len(s1_addr), len(cand_addr)) / max_addr_len
        addr_jaccard, addr_common_cnt = jaccard_similarity(s1["addr_tokens"], cand["addr_tokens"])
        addr_tfidf = addr_tfidf_sims[i]
        addr_num_overlap = numeric_overlap_similarity(s1_addr, cand_addr)

        # 3. Cross-Field & Combined
        comb_tfidf = comb_tfidf_sims[i]
        country_exact = 1.0 if (s1_country_raw and cand_country_raw and s1_country_raw == cand_country_raw) else 0.0
        country_norm = 1.0 if (s1_country_norm and cand_country_norm and s1_country_norm == cand_country_norm) else (0.5 if not s1_country_norm or not cand_country_norm else 0.0)
        country_conf = 1.0 if (s1_country_norm and cand_country_norm and s1_country_norm != cand_country_norm) else 0.0
        comb_weighted = 0.60 * name_wratio + 0.40 * addr_wratio
        total_shared_tok = name_common_cnt + addr_common_cnt
        is_s3 = 1.0 if cand_ids[i].startswith("S3-") else 0.0

        # 4. Missing Indicators
        s1_n_miss = 1.0 if not s1_name else 0.0
        c_n_miss = 1.0 if not cand_name else 0.0
        s1_a_miss = 1.0 if not s1_addr else 0.0
        c_a_miss = 1.0 if not cand_addr else 0.0
        s1_c_miss = 1.0 if not s1_country_norm else 0.0
        c_c_miss = 1.0 if not cand_country_norm else 0.0

        features[i] = [
            name_ratio,
            name_wratio,
            name_qratio,
            name_sort,
            name_set,
            name_partial,
            name_len_diff,
            name_len_ratio,
            name_jaccard,
            name_common_cnt,
            name_tfidf,
            name_prefix_match,
            addr_ratio,
            addr_wratio,
            addr_qratio,
            addr_sort,
            addr_set,
            addr_partial,
            addr_len_diff,
            addr_len_ratio,
            addr_jaccard,
            addr_common_cnt,
            addr_tfidf,
            addr_num_overlap,
            comb_tfidf,
            country_exact,
            country_norm,
            country_conf,
            comb_weighted,
            total_shared_tok,
            is_s3,
            s1_n_miss,
            c_n_miss,
            s1_a_miss,
            c_a_miss,
            s1_c_miss,
            c_c_miss,
        ]

    y = None
    if true_pairs_set is not None:
        y = np.array([1 if (s1_ids[i], cand_ids[i]) in true_pairs_set else 0 for i in range(n_pairs)], dtype=np.int32)

    return df_pairs.copy(), features, y, feature_names
