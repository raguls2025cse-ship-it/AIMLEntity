"""
analyze_blocking_errors.py
Deep diagnostic analysis of the true matches missed by the baseline blocking step.
Extracts the exact missed true pairs, analyzes name/address/country differences,
computes similarity distributions, categorizes failure modes, and saves blocking_error_analysis.csv.
"""

import os
from collections import defaultdict
from typing import Dict, List, Set, Tuple, Any
import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer

from preprocessing import preprocess_dataframe, tokenize_text


def analyze_blocking_failures(
    dev_dir: str = "development/5000_dataset",
    baseline_cand_file: str = "results/baseline/candidate_pairs.tsv",
    output_file: str = "development/results/blocking_error_analysis.csv",
):
    print("=" * 70)
    print("STEP 2: ANALYZING BLOCKING FAILURES (1,841 MISSED TRUE MATCHES)")
    print("=" * 70)

    # Load datasets
    df_s1 = pd.read_csv(os.path.join(dev_dir, "train_source1.tsv"), sep="\t")
    df_s2 = pd.read_csv(os.path.join(dev_dir, "train_source2.tsv"), sep="\t")
    df_s3 = pd.read_csv(os.path.join(dev_dir, "train_source3.tsv"), sep="\t")
    df_gt = pd.read_csv(os.path.join(dev_dir, "train_ground_truth.tsv"), sep="\t")

    print(f"Loaded: S1={len(df_s1):,}, S2={len(df_s2):,}, S3={len(df_s3):,}, GT={len(df_gt):,}")

    # Preprocess
    df_s1_proc = preprocess_dataframe(df_s1)
    df_s2_proc = preprocess_dataframe(df_s2)
    df_s3_proc = preprocess_dataframe(df_s3)

    s1_dict = {row["entity_id"]: row.to_dict() for _, row in df_s1_proc.iterrows()}
    cand_dict = {}
    for _, row in pd.concat([df_s2_proc, df_s3_proc], ignore_index=True).iterrows():
        cand_dict[row["entity_id"]] = row.to_dict()

    # Load baseline candidate pairs
    df_base_cand = pd.read_csv(baseline_cand_file, sep="\t")
    base_cand_set = set(zip(df_base_cand["source1_entity_id"], df_base_cand["candidate_entity_id"]))
    print(f"Baseline candidate pairs loaded: {len(base_cand_set):,}")

    # Extract all true pairs from GT
    all_true_pairs = []
    for _, row in df_gt.iterrows():
        s1_id = str(row["source1_entity_id"]).strip()
        raw_matches = row["matched_entity_ids"]
        if pd.isna(raw_matches) or not str(raw_matches).strip():
            continue
        matches = [m.strip() for m in str(raw_matches).split(",") if m.strip()]
        for m in matches:
            all_true_pairs.append((s1_id, m))

    total_true = len(all_true_pairs)
    missed_pairs = [p for p in all_true_pairs if p not in base_cand_set]
    captured_pairs = [p for p in all_true_pairs if p in base_cand_set]

    print(f"Total True Pairs: {total_true:,}")
    print(f"Captured by Baseline Blocking: {len(captured_pairs):,} ({len(captured_pairs)/total_true*100:.2f}%)")
    print(f"Missed by Baseline Blocking: {len(missed_pairs):,} ({len(missed_pairs)/total_true*100:.2f}%)")

    # Fit TF-IDF models to analyze similarity of missed pairs
    all_names = list(df_s1_proc["name_norm"]) + list(df_s2_proc["name_norm"]) + list(df_s3_proc["name_norm"])
    all_addrs = list(df_s1_proc["address_norm"]) + list(df_s2_proc["address_norm"]) + list(df_s3_proc["address_norm"])
    all_comb = [f"{n} {a}".strip() for n, a in zip(all_names, all_addrs)]

    tfidf_name = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 4), min_df=1, sublinear_tf=True)
    tfidf_name.fit(all_names)

    tfidf_name_char24 = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=1, sublinear_tf=True)
    tfidf_name_char24.fit(all_names)

    tfidf_addr = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 4), min_df=1, sublinear_tf=True)
    tfidf_addr.fit(all_addrs)

    tfidf_comb = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 4), min_df=1, sublinear_tf=True)
    tfidf_comb.fit(all_comb)

    # Analyze missed pairs
    records = []
    category_counts = defaultdict(int)

    for s1_id, cand_id in missed_pairs:
        s1 = s1_dict.get(s1_id, {})
        cand = cand_dict.get(cand_id, {})

        if not s1 or not cand:
            continue

        s1_name_raw = str(s1.get("business_name", ""))
        cand_name_raw = str(cand.get("business_name", ""))
        s1_name_norm = str(s1.get("name_norm", ""))
        cand_name_norm = str(cand.get("name_norm", ""))

        s1_addr_raw = str(s1.get("business_address", ""))
        cand_addr_raw = str(cand.get("business_address", ""))
        s1_addr_norm = str(s1.get("address_norm", ""))
        cand_addr_norm = str(cand.get("address_norm", ""))

        s1_country = str(s1.get("country_norm", ""))
        cand_country = str(cand.get("country_norm", ""))

        s1_name_tokens = set(s1.get("name_tokens", []))
        cand_name_tokens = set(cand.get("name_tokens", []))
        shared_name_tokens = s1_name_tokens.intersection(cand_name_tokens)

        s1_addr_tokens = set(s1.get("addr_tokens", []))
        cand_addr_tokens = set(cand.get("addr_tokens", []))
        shared_addr_tokens = s1_addr_tokens.intersection(cand_addr_tokens)

        # Similarities
        name_ratio = fuzz.ratio(s1_name_norm, cand_name_norm)
        name_wratio = fuzz.WRatio(s1_name_norm, cand_name_norm)
        name_token_sort = fuzz.token_sort_ratio(s1_name_norm, cand_name_norm)
        name_partial = fuzz.partial_ratio(s1_name_norm, cand_name_norm)

        addr_ratio = fuzz.ratio(s1_addr_norm, cand_addr_norm) if s1_addr_norm and cand_addr_norm else 0
        addr_wratio = fuzz.WRatio(s1_addr_norm, cand_addr_norm) if s1_addr_norm and cand_addr_norm else 0
        addr_token_sort = fuzz.token_sort_ratio(s1_addr_norm, cand_addr_norm) if s1_addr_norm and cand_addr_norm else 0

        # TF-IDF Cosine
        v_s1_name = tfidf_name.transform([s1_name_norm])
        v_cand_name = tfidf_name.transform([cand_name_norm])
        name_tfidf_cos = float((v_s1_name @ v_cand_name.T).toarray()[0, 0])

        v_s1_n24 = tfidf_name_char24.transform([s1_name_norm])
        v_cand_n24 = tfidf_name_char24.transform([cand_name_norm])
        name_tfidf_24_cos = float((v_s1_n24 @ v_cand_n24.T).toarray()[0, 0])

        v_s1_addr = tfidf_addr.transform([s1_addr_norm])
        v_cand_addr = tfidf_addr.transform([cand_addr_norm])
        addr_tfidf_cos = float((v_s1_addr @ v_cand_addr.T).toarray()[0, 0])

        comb_s1 = f"{s1_name_norm} {s1_addr_norm}".strip()
        comb_cand = f"{cand_name_norm} {cand_addr_norm}".strip()
        v_s1_comb = tfidf_comb.transform([comb_s1])
        v_cand_comb = tfidf_comb.transform([comb_cand])
        comb_tfidf_cos = float((v_s1_comb @ v_cand_comb.T).toarray()[0, 0])

        # Diagnose primary failure reason
        reason = "Other"
        if not s1_name_norm or not cand_name_norm:
            reason = "Missing/Empty Name"
        elif len(shared_name_tokens) > 0 and name_tfidf_cos < 0.20:
            reason = "High Token Overlap but Capped / TFIDF Under Threshold"
        elif len(shared_addr_tokens) >= 2 and len(shared_name_tokens) == 0:
            reason = "Shared Address Tokens but Distinct Name Tokens (e.g. DBA/Abbrev)"
        elif name_partial >= 80 and len(shared_name_tokens) == 0:
            reason = "Partial/Substring Name Match (no exact whole word overlap)"
        elif name_tfidf_24_cos >= 0.20 and name_tfidf_cos < 0.18:
            reason = "Short/Abbreviated Name caught by 2-4 Char Ngrams"
        elif comb_tfidf_cos >= 0.25 and name_tfidf_cos < 0.18:
            reason = "Combined Name+Address High Cosine (Weak Name Only)"
        elif addr_tfidf_cos >= 0.35:
            reason = "High Address Similarity"
        else:
            reason = "Extreme Variation / Heavy Noise"

        category_counts[reason] += 1

        records.append({
            "source1_entity_id": s1_id,
            "candidate_entity_id": cand_id,
            "s1_name_raw": s1_name_raw,
            "cand_name_raw": cand_name_raw,
            "s1_name_norm": s1_name_norm,
            "cand_name_norm": cand_name_norm,
            "s1_addr_raw": s1_addr_raw,
            "cand_addr_raw": cand_addr_raw,
            "s1_addr_norm": s1_addr_norm,
            "cand_addr_norm": cand_addr_norm,
            "s1_country": s1_country,
            "cand_country": cand_country,
            "name_ratio": name_ratio,
            "name_wratio": name_wratio,
            "name_token_sort": name_token_sort,
            "name_partial": name_partial,
            "addr_wratio": addr_wratio,
            "name_tfidf_34_cos": round(name_tfidf_cos, 4),
            "name_tfidf_24_cos": round(name_tfidf_24_cos, 4),
            "addr_tfidf_34_cos": round(addr_tfidf_cos, 4),
            "comb_tfidf_34_cos": round(comb_tfidf_cos, 4),
            "shared_name_tokens_count": len(shared_name_tokens),
            "shared_addr_tokens_count": len(shared_addr_tokens),
            "primary_failure_reason": reason,
        })

    df_missed = pd.DataFrame(records)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df_missed.to_csv(output_file, index=False)
    print(f"\nSaved detailed error analysis for {len(df_missed):,} missed pairs to {output_file}")

    print("\n" + "=" * 70)
    print("MISSED TRUE PAIRS BREAKDOWN BY ROOT CAUSE:")
    for reason, count in sorted(category_counts.items(), key=lambda x: -x[1]):
        pct = (count / len(df_missed)) * 100
        print(f"  - {reason:<60}: {count:>4} ({pct:5.1f}%)")
    print("=" * 70)

    # Key statistics of missed pairs
    print("\nRECOVERABILITY POTENTIAL WITH COMPLEMENTARY STRATEGIES:")
    rec_by_comb_tfidf = (df_missed["comb_tfidf_34_cos"] >= 0.15).sum()
    rec_by_addr_tfidf = (df_missed["addr_tfidf_34_cos"] >= 0.25).sum()
    rec_by_name_24 = (df_missed["name_tfidf_24_cos"] >= 0.15).sum()
    rec_by_addr_tokens = (df_missed["shared_addr_tokens_count"] >= 2).sum()
    rec_by_partial_name = (df_missed["name_partial"] >= 75).sum()
    
    print(f"  - Can be recovered by Combined Name+Address TF-IDF (sim >= 0.15): {rec_by_comb_tfidf:,} ({rec_by_comb_tfidf/len(df_missed)*100:.1f}%)")
    print(f"  - Can be recovered by 2-4 Char Ngram Name TF-IDF (sim >= 0.15):    {rec_by_name_24:,} ({rec_by_name_24/len(df_missed)*100:.1f}%)")
    print(f"  - Can be recovered by Address TF-IDF / Token Blocking:             {rec_by_addr_tfidf:,} ({rec_by_addr_tfidf/len(df_missed)*100:.1f}%)")
    print(f"  - Can be recovered by Partial Name Match (partial_ratio >= 75):     {rec_by_partial_name:,} ({rec_by_partial_name/len(df_missed)*100:.1f}%)")
    print("=" * 70 + "\n")

    return df_missed


if __name__ == "__main__":
    analyze_blocking_failures()
