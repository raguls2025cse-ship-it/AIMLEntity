"""
run_training.py
Main end-to-end training and validation script for Entity Matching.
Orchestrates dataset verification, preprocessing, improved multi-strategy blocking,
rich feature engineering, entity-grouped training, threshold tuning, and artifact generation.
Generates baseline vs improved comparison.
"""

import os
import sys
import json
import joblib
from collections import defaultdict
from typing import Dict, List, Set, Tuple, Any
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from create_5000_dataset import create_development_dataset
from preprocessing import preprocess_dataframe
from blocking import generate_candidate_pairs, evaluate_blocking_recall
from features import build_vectorizers, extract_features_for_pairs
from train_model import train_and_evaluate_models
from evaluate import evaluate_entity_macro_f05


def main():
    print("=" * 70, flush=True)
    print("      AIML HACKATHON: BUSINESS ENTITY RESOLUTION PIPELINE", flush=True)
    print("      IMPROVED 5,000-SOURCE-1 TRAINING & VALIDATION", flush=True)
    print("=" * 70, flush=True)

    # Baseline constants for reference
    BASELINE_BLOCKING_RECALL = 0.8941
    BASELINE_CANDIDATE_COUNT = 260500
    BASELINE_MACRO_PRECISION = 0.9772
    BASELINE_MACRO_RECALL = 0.8890
    BASELINE_MACRO_F05 = 0.9472
    BASELINE_THRESHOLD = 0.70

    # Paths
    raw_train_dir = "train"
    dev_dir = "development/5000_dataset"
    results_dir = "development/results"
    models_dir = "models"
    improved_models_dir = "models/improved"

    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(improved_models_dir, exist_ok=True)

    # 1. Verification and Dataset Loading
    s1_file = os.path.join(dev_dir, "train_source1.tsv")
    s2_file = os.path.join(dev_dir, "train_source2.tsv")
    s3_file = os.path.join(dev_dir, "train_source3.tsv")
    gt_file = os.path.join(dev_dir, "train_ground_truth.tsv")

    if not (os.path.exists(s1_file) and os.path.exists(s2_file) and os.path.exists(s3_file) and os.path.exists(gt_file)):
        print("\n[INFO] Development dataset not found. Creating 5,000 S1 dataset from raw files...", flush=True)
        create_development_dataset(
            train_dir=raw_train_dir,
            output_dir=dev_dir,
            n_s1=5000,
            n_negative_s2=5000,
            n_negative_s3=5000,
            random_seed=42,
        )
    else:
        print("\n[INFO] Found existing 5,000 S1 development dataset.", flush=True)

    print("\nLoading development dataset files...", flush=True)
    df_s1 = pd.read_csv(s1_file, sep="\t")
    df_s2 = pd.read_csv(s2_file, sep="\t")
    df_s3 = pd.read_csv(s3_file, sep="\t")
    df_gt = pd.read_csv(gt_file, sep="\t")

    print(f"Loaded records: S1={len(df_s1):,}, S2={len(df_s2):,}, S3={len(df_s3):,}, GT={len(df_gt):,}", flush=True)

    # Ground Truth Map
    gt_matches_dict = defaultdict(set)
    all_true_pairs_set = set()
    num_singletons = 0
    total_pos_pairs = 0

    for _, row in df_gt.iterrows():
        s1_id = str(row["source1_entity_id"]).strip()
        raw_matches = row["matched_entity_ids"]
        if pd.isna(raw_matches) or not str(raw_matches).strip():
            num_singletons += 1
            gt_matches_dict[s1_id] = set()
            continue
        matches = [m.strip() for m in str(raw_matches).split(",") if m.strip()]
        if not matches:
            num_singletons += 1
            gt_matches_dict[s1_id] = set()
            continue
        for m in matches:
            gt_matches_dict[s1_id].add(m)
            all_true_pairs_set.add((s1_id, m))
            total_pos_pairs += 1

    print(f"Ground Truth Analysis: {total_pos_pairs:,} positive pairs, {num_singletons:,} singletons.", flush=True)

    # 2. Preprocessing
    print("\nPreprocessing entity records (safe normalization & tokenization)...", flush=True)
    df_s1_proc = preprocess_dataframe(df_s1)
    df_s2_proc = preprocess_dataframe(df_s2)
    df_s3_proc = preprocess_dataframe(df_s3)

    s1_dict = {row["entity_id"]: row.to_dict() for _, row in df_s1_proc.iterrows()}
    cand_dict = {}
    for _, row in pd.concat([df_s2_proc, df_s3_proc], ignore_index=True).iterrows():
        cand_dict[row["entity_id"]] = row.to_dict()

    # 3. Improved Multi-Strategy Blocking & Candidate Generation
    df_candidate_pairs = generate_candidate_pairs(
        df_s1=df_s1_proc,
        df_s2=df_s2_proc,
        df_s3=df_s3_proc,
        top_k_comb=35,
        min_sim_comb=0.16,
        top_k_name=25,
        min_sim_name=0.20,
        top_k_addr=20,
        min_sim_addr=0.35,
    )

    # Save Candidate Pairs
    cand_pairs_path = os.path.join(results_dir, "candidate_pairs.tsv")
    df_candidate_pairs.to_csv(cand_pairs_path, sep="\t", index=False)
    print(f"Saved candidate pairs to {cand_pairs_path}", flush=True)

    # Evaluate Blocking Recall
    blocking_metrics = evaluate_blocking_recall(df_candidate_pairs, df_gt)

    # 4. Feature Engineering
    print("\nBuilding TF-IDF vectorizers on names, addresses, and combined representations...", flush=True)
    all_names = list(df_s1_proc["name_norm"]) + list(df_s2_proc["name_norm"]) + list(df_s3_proc["name_norm"])
    all_addrs = list(df_s1_proc["address_norm"]) + list(df_s2_proc["address_norm"]) + list(df_s3_proc["address_norm"])
    all_comb = [f"{n} {a}".strip() for n, a in zip(all_names, all_addrs)]
    name_vectorizer, addr_vectorizer, comb_vectorizer = build_vectorizers(all_names, all_addrs, all_comb)

    # Extract Features for All Candidate Pairs
    df_pairs_info, X, y, feature_names = extract_features_for_pairs(
        df_pairs=df_candidate_pairs,
        s1_dict=s1_dict,
        cand_dict=cand_dict,
        name_vec=name_vectorizer,
        addr_vec=addr_vectorizer,
        comb_vec=comb_vectorizer,
        true_pairs_set=all_true_pairs_set,
    )

    df_pairs_info["is_true_match"] = y
    num_neg_pairs = int((y == 0).sum())
    num_pos_pairs_in_cand = int((y == 1).sum())
    print(f"Candidate Pair Labels: Positive={num_pos_pairs_in_cand:,}, Negative={num_neg_pairs:,}", flush=True)

    # 5. Entity-Grouped Train / Validation Split (Seed 42)
    print("\nSplitting 5,000 Source-1 entities into Train (80%) and Validation (20%)...", flush=True)
    unique_s1_ids = np.array(list(df_s1["entity_id"].unique()))
    train_s1_ids, val_s1_ids = train_test_split(unique_s1_ids, test_size=0.20, random_state=42)
    train_s1_set = set(train_s1_ids)
    val_s1_set = set(val_s1_ids)

    print(f"Train S1 Entities: {len(train_s1_set):,} | Validation S1 Entities: {len(val_s1_set):,}", flush=True)

    train_mask = df_pairs_info["source1_entity_id"].isin(train_s1_set).values
    val_mask = df_pairs_info["source1_entity_id"].isin(val_s1_set).values

    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    val_pairs_df = df_pairs_info[val_mask].reset_index(drop=True)

    # 6. Train Models & Compare
    best_model_name, best_model, best_threshold, best_metrics, df_comparison, df_all_thresh = train_and_evaluate_models(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        val_pairs_df=val_pairs_df,
        val_s1_ids=list(val_s1_ids),
        gt_matches_dict=gt_matches_dict,
        feature_names=feature_names,
        random_seed=42,
    )

    # 7. Save Threshold Sweep Results
    thresh_results_path = os.path.join(results_dir, "threshold_results.csv")
    df_all_thresh.to_csv(thresh_results_path, index=False)
    print(f"Saved threshold sweep results to {thresh_results_path}", flush=True)

    # 8. Save Validation Predictions
    val_probs = best_model.predict_proba(X_val)[:, 1]
    val_predictions_df = pd.DataFrame({
        "source1_entity_id": val_pairs_df["source1_entity_id"],
        "candidate_entity_id": val_pairs_df["candidate_entity_id"],
        "prediction_probability": val_probs.round(6),
        "predicted_match": (val_probs >= best_threshold).astype(int),
        "true_match": y_val,
    })
    val_pred_path = os.path.join(results_dir, "validation_predictions.tsv")
    val_predictions_df.to_csv(val_pred_path, sep="\t", index=False)
    print(f"Saved validation predictions to {val_pred_path}", flush=True)

    # 9. Save Models and Config (Both in models/ and models/improved/)
    for target_dir in [models_dir, improved_models_dir]:
        joblib.dump(best_model, os.path.join(target_dir, "entity_matcher.pkl"))
        joblib.dump(name_vectorizer, os.path.join(target_dir, "name_vectorizer.pkl"))
        joblib.dump(addr_vectorizer, os.path.join(target_dir, "address_vectorizer.pkl"))
        joblib.dump(comb_vectorizer, os.path.join(target_dir, "comb_vectorizer.pkl"))

        model_config = {
            "model_name": best_model_name,
            "best_threshold": float(best_threshold),
            "features": feature_names,
            "validation_metrics": best_metrics,
            "blocking_metrics": blocking_metrics,
            "train_s1_count": len(train_s1_set),
            "val_s1_count": len(val_s1_set),
            "random_seed": 42,
        }

        with open(os.path.join(target_dir, "model_config.json"), "w") as f:
            json.dump(model_config, f, indent=2)

    # Save Model Metrics
    metrics_path = os.path.join(results_dir, "model_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump({
            "model_comparison": df_comparison.to_dict(orient="records"),
            "winning_model": best_model_name,
            "best_threshold": best_threshold,
            "validation_metrics": best_metrics,
            "blocking_metrics": blocking_metrics,
        }, f, indent=2)
    print(f"Saved model metrics to {metrics_path}", flush=True)

    # 10. Baseline vs Improved Comparison Table
    df_base_vs_imp = pd.DataFrame([
        {
            "Metric": "Blocking Recall",
            "Baseline": f"{BASELINE_BLOCKING_RECALL*100:.2f}%",
            "Improved": f"{blocking_metrics['blocking_recall']*100:.2f}%",
            "Delta": f"{(blocking_metrics['blocking_recall'] - BASELINE_BLOCKING_RECALL)*100:+.2f}%",
        },
        {
            "Metric": "Candidate Pairs",
            "Baseline": f"{BASELINE_CANDIDATE_COUNT:,}",
            "Improved": f"{len(df_candidate_pairs):,}",
            "Delta": f"{len(df_candidate_pairs) - BASELINE_CANDIDATE_COUNT:+,}",
        },
        {
            "Metric": "Validation Macro Precision",
            "Baseline": f"{BASELINE_MACRO_PRECISION*100:.2f}%",
            "Improved": f"{best_metrics['macro_precision']*100:.2f}%",
            "Delta": f"{(best_metrics['macro_precision'] - BASELINE_MACRO_PRECISION)*100:+.2f}%",
        },
        {
            "Metric": "Validation Macro Recall",
            "Baseline": f"{BASELINE_MACRO_RECALL*100:.2f}%",
            "Improved": f"{best_metrics['macro_recall']*100:.2f}%",
            "Delta": f"{(best_metrics['macro_recall'] - BASELINE_MACRO_RECALL)*100:+.2f}%",
        },
        {
            "Metric": "Validation Macro F0.5",
            "Baseline": f"{BASELINE_MACRO_F05:.4f}",
            "Improved": f"{best_metrics['macro_f05']:.4f}",
            "Delta": f"{best_metrics['macro_f05'] - BASELINE_MACRO_F05:+.4f}",
        },
        {
            "Metric": "Optimal Threshold",
            "Baseline": f"{BASELINE_THRESHOLD:.2f}",
            "Improved": f"{best_threshold:.2f}",
            "Delta": f"{best_threshold - BASELINE_THRESHOLD:+.2f}",
        },
    ])

    base_vs_imp_path = os.path.join(results_dir, "baseline_vs_improved.csv")
    df_base_vs_imp.to_csv(base_vs_imp_path, index=False)
    print(f"Saved baseline vs improved comparison to {base_vs_imp_path}", flush=True)

    # 11. Final Summary Report
    print("\n" + "=" * 70, flush=True)
    print("                    FINAL IMPROVED SUMMARY REPORT", flush=True)
    print("=" * 70, flush=True)
    print(f"1. Baseline Blocking Recall:              {BASELINE_BLOCKING_RECALL*100:.2f}%", flush=True)
    print(f"2. Improved Blocking Recall:              {blocking_metrics['blocking_recall']*100:.2f}% ({blocking_metrics['captured_true_matches']:,}/{blocking_metrics['total_true_matches']:,})", flush=True)
    print(f"3. Baseline Candidate Count:              {BASELINE_CANDIDATE_COUNT:,}", flush=True)
    print(f"4. Improved Candidate Count:              {len(df_candidate_pairs):,}", flush=True)
    print(f"5. Baseline Macro Precision:              {BASELINE_MACRO_PRECISION*100:.2f}%", flush=True)
    print(f"6. Improved Macro Precision:              {best_metrics['macro_precision']*100:.2f}%", flush=True)
    print(f"7. Baseline Macro Recall:                 {BASELINE_MACRO_RECALL*100:.2f}%", flush=True)
    print(f"8. Improved Macro Recall:                 {best_metrics['macro_recall']*100:.2f}%", flush=True)
    print(f"9. Baseline Macro F0.5:                   {BASELINE_MACRO_F05:.4f}", flush=True)
    print(f"10. Improved Macro F0.5:                  {best_metrics['macro_f05']:.4f}", flush=True)
    print(f"11. Best Decision Threshold:              {best_threshold:.2f}", flush=True)
    print("12. Best Blocking Strategies:             Exact Name + Name Tokens + Name TFIDF + Address Tokens + Combined TFIDF + Address TFIDF", flush=True)
    print(f"13. Candidate Count per S1 (Average):     {len(df_candidate_pairs)/len(df_s1):.1f}", flush=True)
    print("14. Model Comparison (Entity Macro F0.5):", flush=True)
    for _, r in df_comparison.iterrows():
        print(f"   - {r['model_name']:<22}: Macro F0.5 = {r['macro_f05']:.4f} | Prec = {r['macro_precision']:.4f} | Rec = {r['macro_recall']:.4f} (Thresh = {r['best_threshold']:.2f})", flush=True)
    print(f"15. Replace Baseline Recommendation:      {'YES - Statistically superior across Recall & Macro F0.5' if best_metrics['macro_f05'] >= BASELINE_MACRO_F05 else 'NO - Keep Baseline'}", flush=True)
    print("=" * 70, flush=True)


if __name__ == "__main__":
    main()
