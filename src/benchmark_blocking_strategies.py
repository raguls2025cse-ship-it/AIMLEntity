"""
benchmark_blocking_strategies.py
Fast, vectorized benchmarking of individual and combined blocking strategies.
Computes recall, candidate counts, reduction ratio, and saves blocking_strategy_comparison.csv.
"""

import os
import sys
from collections import defaultdict
from typing import Dict, List, Set, Tuple, Any
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from preprocessing import preprocess_dataframe, tokenize_text


COMMON_NAME_STOPWORDS = {
    "inc", "corp", "corporation", "incorporated", "llc", "ltd", "limited", "co",
    "company", "pvt", "gmbh", "sa", "llp", "plc", "the", "and", "of", "for", "in",
    "at", "by", "a", "an", "group", "ent", "enterprises", "svc", "services",
    "tech", "technologies", "intl", "international", "assoc", "associates"
}

COMMON_ADDR_STOPWORDS = {
    "st", "rd", "ave", "blvd", "dr", "ln", "ct", "pl", "pkwy", "hwy", "ste", "apt",
    "fl", "bldg", "pobox", "box", "n", "s", "e", "w", "ne", "nw", "se", "sw",
    "the", "and", "of", "in", "at", "by", "road", "street", "avenue", "suite"
}


def batch_topk_sparse(s1_mat, cand_mat, top_k: int, min_sim: float, batch_size: int = 500):
    """Fast batch sparse dot product and top-K candidate extraction."""
    n_s1 = s1_mat.shape[0]
    results = [set() for _ in range(n_s1)]
    
    for start_idx in range(0, n_s1, batch_size):
        end_idx = min(start_idx + batch_size, n_s1)
        sim_batch = s1_mat[start_idx:end_idx].dot(cand_mat.T).tocsr()
        
        for local_i in range(end_idx - start_idx):
            global_i = start_idx + local_i
            row_start = sim_batch.indptr[local_i]
            row_end = sim_batch.indptr[local_i + 1]
            
            if row_end > row_start:
                row_cols = sim_batch.indices[row_start:row_end]
                row_data = sim_batch.data[row_start:row_end]
                
                valid_mask = row_data >= min_sim
                if np.any(valid_mask):
                    v_cols = row_cols[valid_mask]
                    v_data = row_data[valid_mask]
                    
                    if len(v_cols) > top_k:
                        top_order = np.argpartition(-v_data, top_k)[:top_k]
                        selected = v_cols[top_order]
                    else:
                        selected = v_cols
                    results[global_i].update(selected)
                    
    return results


def benchmark_strategies(
    dev_dir: str = "development/5000_dataset",
    output_comparison_file: str = "development/results/blocking_strategy_comparison.csv",
):
    print("=" * 70, flush=True)
    print("STEP 3-6: BENCHMARKING BLOCKING STRATEGIES & UNION EXPERIMENTS", flush=True)
    print("=" * 70, flush=True)

    df_s1 = pd.read_csv(os.path.join(dev_dir, "train_source1.tsv"), sep="\t")
    df_s2 = pd.read_csv(os.path.join(dev_dir, "train_source2.tsv"), sep="\t")
    df_s3 = pd.read_csv(os.path.join(dev_dir, "train_source3.tsv"), sep="\t")
    df_gt = pd.read_csv(os.path.join(dev_dir, "train_ground_truth.tsv"), sep="\t")

    df_s1_proc = preprocess_dataframe(df_s1)
    df_s2_proc = preprocess_dataframe(df_s2)
    df_s3_proc = preprocess_dataframe(df_s3)

    df_cand = pd.concat([df_s2_proc, df_s3_proc], ignore_index=True)
    idx_to_cand_id = df_cand["entity_id"].values
    cand_names = df_cand["name_norm"].values
    cand_addrs = df_cand["address_norm"].values
    cand_comb = [f"{n} {a}".strip() for n, a in zip(cand_names, cand_addrs)]
    cand_countries = df_cand["country_norm"].values

    s1_names = df_s1_proc["name_norm"].values
    s1_addrs = df_s1_proc["address_norm"].values
    s1_comb = [f"{n} {a}".strip() for n, a in zip(s1_names, s1_addrs)]
    s1_countries = df_s1_proc["country_norm"].values

    total_possible_pairs = len(df_s1) * len(df_cand)

    true_pairs = set()
    for _, row in df_gt.iterrows():
        s1_id = str(row["source1_entity_id"]).strip()
        raw = row["matched_entity_ids"]
        if pd.isna(raw) or not str(raw).strip():
            continue
        for m in str(raw).split(","):
            if m.strip():
                true_pairs.add((s1_id, m.strip()))

    total_true = len(true_pairs)
    print(f"Total True Pairs in Ground Truth: {total_true:,}", flush=True)

    # Inverted Indices
    print("\n1. Building Exact Name Map...", flush=True)
    exact_name_map = defaultdict(list)
    for idx, name in enumerate(cand_names):
        if name.strip():
            exact_name_map[name.strip()].append(idx)

    print("2. Building Name Token Inverted Index...", flush=True)
    name_token_idx = defaultdict(list)
    for idx, tokens in enumerate(df_cand["name_tokens"]):
        seen = set()
        for token in tokens:
            if len(token) >= 3 and token not in COMMON_NAME_STOPWORDS and token not in seen:
                name_token_idx[token].append(idx)
                seen.add(token)

    print("3. Building Address Token Inverted Index...", flush=True)
    addr_token_idx = defaultdict(list)
    for idx, tokens in enumerate(df_cand["addr_tokens"]):
        seen = set()
        for token in tokens:
            if len(token) >= 3 and token not in COMMON_ADDR_STOPWORDS and token not in seen:
                addr_token_idx[token].append(idx)
                seen.add(token)

    print("4. Fitting TF-IDF Vectorizers...", flush=True)
    tfidf_name_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 4), min_df=1, sublinear_tf=True)
    cand_name_mat = tfidf_name_vec.fit_transform(cand_names)
    s1_name_mat = tfidf_name_vec.transform(s1_names)

    tfidf_comb_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 4), min_df=1, sublinear_tf=True)
    cand_comb_mat = tfidf_comb_vec.fit_transform(cand_comb)
    s1_comb_mat = tfidf_comb_vec.transform(s1_comb)

    tfidf_addr_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 4), min_df=1, sublinear_tf=True)
    cand_addr_mat = tfidf_addr_vec.fit_transform(cand_addrs)
    s1_addr_mat = tfidf_addr_vec.transform(s1_addrs)

    def eval_pair_list(strat_name: str, s1_cand_list: List[Set[int]]) -> Dict[str, Any]:
        pair_set = set()
        cand_counts = []
        for s1_idx, c_set in enumerate(s1_cand_list):
            s1_id = df_s1_proc.iloc[s1_idx]["entity_id"]
            cand_counts.append(len(c_set))
            for c_idx in c_set:
                pair_set.add((s1_id, idx_to_cand_id[c_idx]))
        
        captured = len(true_pairs.intersection(pair_set))
        rec = captured / total_true if total_true > 0 else 0
        cand_count = len(pair_set)
        reduction = 1.0 - (cand_count / total_possible_pairs)

        return {
            "strategy": strat_name,
            "candidate_count": cand_count,
            "captured_true_matches": captured,
            "blocking_recall": rec,
            "blocking_recall_pct": f"{rec*100:.2f}%",
            "avg_candidates_per_s1": float(np.mean(cand_counts)) if cand_counts else 0,
            "max_candidates_per_s1": int(np.max(cand_counts)) if cand_counts else 0,
            "reduction_ratio": f"{reduction*100:.4f}%",
            "cand_list": s1_cand_list,
        }

    results = []

    # Strategy 1: Exact Name
    print("\nEvaluating Strategy 1: Exact Name Matching...", flush=True)
    s1_strat1 = [set() for _ in range(len(df_s1_proc))]
    for s1_idx, name in enumerate(s1_names):
        if name in exact_name_map:
            s1_strat1[s1_idx].update(exact_name_map[name])
    r1 = eval_pair_list("1. Exact Normalized Name", s1_strat1)
    results.append(r1)

    # Strategy 2: Name Token Inverted Index
    print("Evaluating Strategy 2: Name Token Inverted Index...", flush=True)
    s1_strat2 = [set() for _ in range(len(df_s1_proc))]
    for s1_idx, tokens in enumerate(df_s1_proc["name_tokens"]):
        for t in tokens:
            if len(t) >= 3 and t not in COMMON_NAME_STOPWORDS:
                match_indices = name_token_idx.get(t, [])
                if 0 < len(match_indices) <= 250:
                    s1_strat2[s1_idx].update(match_indices)
    r2 = eval_pair_list("2. Name Token Inverted Index", s1_strat2)
    results.append(r2)

    # Strategy 3: Name TF-IDF Retrieval
    print("Evaluating Strategy 3: Name TF-IDF Retrieval...", flush=True)
    s1_strat3 = batch_topk_sparse(s1_name_mat, cand_name_mat, top_k=25, min_sim=0.20)
    r3 = eval_pair_list("3. Name TF-IDF (top-25, sim>=0.20)", s1_strat3)
    results.append(r3)

    # Strategy 4: Address Token Inverted Index (Country-Aware)
    print("Evaluating Strategy 4: Address Token Inverted Index...", flush=True)
    s1_strat4 = [set() for _ in range(len(df_s1_proc))]
    for s1_idx, tokens in enumerate(df_s1_proc["addr_tokens"]):
        s1_country = s1_countries[s1_idx]
        token_cands = defaultdict(int)
        for t in tokens:
            if len(t) >= 3 and t not in COMMON_ADDR_STOPWORDS:
                m_list = addr_token_idx.get(t, [])
                if 0 < len(m_list) <= 150:
                    for c_idx in m_list:
                        c_country = cand_countries[c_idx]
                        if not s1_country or not c_country or s1_country == c_country:
                            token_cands[c_idx] += 1

        for c_idx, cnt in token_cands.items():
            if cnt >= 2:
                s1_strat4[s1_idx].add(c_idx)
            elif cnt == 1 and len(df_cand.iloc[c_idx]["addr_tokens"]) <= 4:
                s1_strat4[s1_idx].add(c_idx)
    r4 = eval_pair_list("4. Address Token Index (shared>=2 / country-aware)", s1_strat4)
    results.append(r4)

    # Strategy 5: Combined Name+Address TF-IDF Retrieval
    print("Evaluating Strategy 5: Combined Name+Address TF-IDF Retrieval...", flush=True)
    s1_strat5 = batch_topk_sparse(s1_comb_mat, cand_comb_mat, top_k=30, min_sim=0.18)
    r5 = eval_pair_list("5. Combined Name+Address TF-IDF (top-30, sim>=0.18)", s1_strat5)
    results.append(r5)

    # Strategy 6: Address TF-IDF Retrieval (Country-Aware)
    print("Evaluating Strategy 6: Address TF-IDF Retrieval...", flush=True)
    raw_addr_cands = batch_topk_sparse(s1_addr_mat, cand_addr_mat, top_k=20, min_sim=0.35)
    s1_strat6 = [set() for _ in range(len(df_s1_proc))]
    for s1_idx in range(len(df_s1_proc)):
        s1_country = s1_countries[s1_idx]
        for c_idx in raw_addr_cands[s1_idx]:
            c_country = cand_countries[c_idx]
            if not s1_country or not c_country or s1_country == c_country:
                s1_strat6[s1_idx].add(c_idx)
    r6 = eval_pair_list("6. Address TF-IDF (top-20, sim>=0.35, country-aware)", s1_strat6)
    results.append(r6)

    # Strategy 7: Union of All Strategies
    print("\nEvaluating Strategy 7: Full Complementary Union...", flush=True)
    s1_union_all = [set() for _ in range(len(df_s1_proc))]
    for s1_idx in range(len(df_s1_proc)):
        s1_union_all[s1_idx].update(s1_strat1[s1_idx])
        s1_union_all[s1_idx].update(s1_strat2[s1_idx])
        s1_union_all[s1_idx].update(s1_strat3[s1_idx])
        s1_union_all[s1_idx].update(s1_strat4[s1_idx])
        s1_union_all[s1_idx].update(s1_strat5[s1_idx])
        s1_union_all[s1_idx].update(s1_strat6[s1_idx])
    r_union = eval_pair_list("7. FULL UNION (All 6 Strategies)", s1_union_all)
    results.append(r_union)

    # Save Comparison CSV
    df_results = pd.DataFrame([{
        "strategy": r["strategy"],
        "candidate_count": r["candidate_count"],
        "captured_true_matches": r["captured_true_matches"],
        "blocking_recall": round(r["blocking_recall"], 4),
        "blocking_recall_pct": r["blocking_recall_pct"],
        "avg_candidates_per_s1": round(r["avg_candidates_per_s1"], 1),
        "max_candidates_per_s1": r["max_candidates_per_s1"],
        "reduction_ratio": r["reduction_ratio"],
    } for r in results])

    os.makedirs(os.path.dirname(output_comparison_file), exist_ok=True)
    df_results.to_csv(output_comparison_file, index=False)
    print(f"\nSaved blocking strategy comparison to {output_comparison_file}", flush=True)

    print("\n" + "=" * 70, flush=True)
    print("BLOCKING STRATEGIES COMPARISON SUMMARY:", flush=True)
    print("=" * 70, flush=True)
    print(f"{'Strategy':<46} | {'Recall':<8} | {'Candidates':<10} | {'Avg/S1':<6} | {'Reduction':<9}", flush=True)
    print("-" * 70, flush=True)
    for _, r in df_results.iterrows():
        print(f"{r['strategy']:<46} | {r['blocking_recall_pct']:<8} | {r['candidate_count']:<10,d} | {r['avg_candidates_per_s1']:<6.1f} | {r['reduction_ratio']:<9}", flush=True)
    print("=" * 70 + "\n", flush=True)

    return df_results


if __name__ == "__main__":
    benchmark_strategies()
