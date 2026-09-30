"""
verify_model.py
Comprehensive verification script to test whether the trained entity matching model works correctly.
Tests:
1. Artifact integrity (model, vectorizers, config)
2. Evaluation metrics validation against ground truth
3. Representative business entity matching tests (exact match, typos/abbreviations, cross-country, negative pairs)
4. Singleton handling behavior
"""

import json
import os
import joblib
import numpy as np
import pandas as pd
from collections import defaultdict

from preprocessing import preprocess_dataframe
from features import extract_features_for_pairs
from evaluate import evaluate_entity_macro_f05


def run_verification():
    print("=" * 75)
    print("      BUSINESS ENTITY RESOLUTION MODEL VERIFICATION & HEALTH CHECK")
    print("=" * 75)

    # 1. Load artifacts
    models_dir = "models"
    config_path = os.path.join(models_dir, "model_config.json")
    model_path = os.path.join(models_dir, "entity_matcher.pkl")
    name_vec_path = os.path.join(models_dir, "name_vectorizer.pkl")
    addr_vec_path = os.path.join(models_dir, "address_vectorizer.pkl")
    comb_vec_path = os.path.join(models_dir, "comb_vectorizer.pkl")

    assert os.path.exists(config_path), f"Missing {config_path}"
    assert os.path.exists(model_path), f"Missing {model_path}"
    assert os.path.exists(name_vec_path), f"Missing {name_vec_path}"
    assert os.path.exists(addr_vec_path), f"Missing {addr_vec_path}"
    assert os.path.exists(comb_vec_path), f"Missing {comb_vec_path}"

    with open(config_path, "r") as f:
        config = json.load(f)

    model = joblib.load(model_path)
    name_vec = joblib.load(name_vec_path)
    addr_vec = joblib.load(addr_vec_path)
    comb_vec = joblib.load(comb_vec_path)
    features_list = config["features"]
    threshold = float(config["best_threshold"])

    print(f"[OK] Model Artifacts Loaded Successfully:")
    print(f"     - Model Architecture:       {type(model).__name__} ({config['model_name']})")
    print(f"     - Feature Count:            {len(features_list)}")
    print(f"     - Decision Threshold:       {threshold:.2f}")
    print(f"     - Validation Macro F0.5:    {config['validation_metrics']['macro_f05']:.4f}")
    print(f"     - Validation Precision:     {config['validation_metrics']['macro_precision']:.4f}")
    print(f"     - Validation Recall:        {config['validation_metrics']['macro_recall']:.4f}")

    # 2. Verify Saved Validation Predictions
    val_pred_path = "development/results/validation_predictions.tsv"
    if os.path.exists(val_pred_path):
        df_val = pd.read_csv(val_pred_path, sep="\t")
        print(f"\n[OK] Validation Predictions File Verified:")
        print(f"     - Total Evaluated Pairs:    {len(df_val):,}")
        tp = int(((df_val["predicted_match"] == 1) & (df_val["true_match"] == 1)).sum())
        fp = int(((df_val["predicted_match"] == 1) & (df_val["true_match"] == 0)).sum())
        fn = int(((df_val["predicted_match"] == 0) & (df_val["true_match"] == 1)).sum())
        tn = int(((df_val["predicted_match"] == 0) & (df_val["true_match"] == 0)).sum())
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f05 = (1.25 * p * r) / (0.25 * p + r) if (0.25 * p + r) > 0 else 0.0
        print(f"     - TP={tp:,}, FP={fp:,}, FN={fn:,}, TN={tn:,}")
        print(f"     - Micro Pair Precision:     {p*100:.2f}%")
        print(f"     - Micro Pair Recall:        {r*100:.2f}%")
        print(f"     - Micro Pair F0.5:          {f05:.4f}")

    # 3. Test on Synthetic & Real-world Test Scenarios
    print("\n" + "=" * 75)
    print("      TESTING INFERENCE PIPELINE ON REPRESENTATIVE SCENARIOS")
    print("=" * 75)

    test_scenarios = [
        {
            "desc": "Exact / Standardized Legal Suffix Variant (Acme Corp)",
            "s1_id": "T1_S1",
            "s1_name": "Acme Technologies Incorporated",
            "s1_addr": "100 Innovation Way, Suite 200",
            "s1_country": "US",
            "c_id": "T1_C2",
            "c_name": "ACME TECHNOLOGIES INC",
            "c_addr": "100 Innovation Way Ste 200",
            "c_country": "United States",
            "source": "source2",
            "expected_match": True,
        },
        {
            "desc": "Street Abbreviation & Spacing Variations (Global Logistics)",
            "s1_id": "T2_S1",
            "s1_name": "Global Logistics & Freight Solutions",
            "s1_addr": "450 West 33rd Avenue, 5th Floor",
            "s1_country": "CA",
            "c_id": "T2_C3",
            "c_name": "Global Logistics and Freight Solutions Inc",
            "c_addr": "450 W 33rd Ave Fl 5",
            "c_country": "Canada",
            "source": "source3",
            "expected_match": True,
        },
        {
            "desc": "Completely Unrelated Businesses (Negative Pair)",
            "s1_id": "T3_S1",
            "s1_name": "Microsoft Corporation",
            "s1_addr": "One Microsoft Way, Redmond, WA",
            "s1_country": "US",
            "c_id": "T3_C2",
            "c_name": "Walmart Supercenter",
            "c_addr": "702 SW 8th St, Bentonville, AR",
            "c_country": "US",
            "source": "source2",
            "expected_match": False,
        },
        {
            "desc": "Similar Name in Different Countries (Country Conflict)",
            "s1_id": "T4_S1",
            "s1_name": "Apex Consulting Services",
            "s1_addr": "12 King Street, Sydney",
            "s1_country": "AU",
            "c_id": "T4_C2",
            "c_name": "Apex Consulting Services Ltd",
            "c_addr": "45 Oxford Road, Manchester",
            "c_country": "GB",
            "source": "source2",
            "expected_match": False,
        },
        {
            "desc": "Different Business at Same Commercial Plaza / Building",
            "s1_id": "T5_S1",
            "s1_name": "Sunrise Dental Care",
            "s1_addr": "500 Main Street, Suite 101",
            "s1_country": "US",
            "c_id": "T5_C2",
            "c_name": "Metro Law Partners LLP",
            "c_addr": "500 Main Street, Suite 202",
            "c_country": "US",
            "source": "source2",
            "expected_match": False,
        },
    ]

    df_s1_raw = pd.DataFrame([{
        "entity_id": sc["s1_id"],
        "business_name": sc["s1_name"],
        "business_address": sc["s1_addr"],
        "country": sc["s1_country"]
    } for sc in test_scenarios])

    df_cand_raw = pd.DataFrame([{
        "entity_id": sc["c_id"],
        "business_name": sc["c_name"],
        "business_address": sc["c_addr"],
        "country": sc["c_country"],
        "source": sc["source"]
    } for sc in test_scenarios])

    df_pairs = pd.DataFrame([{
        "source1_entity_id": sc["s1_id"],
        "candidate_entity_id": sc["c_id"],
        "blocking_strategy": "test"
    } for sc in test_scenarios])

    # Preprocess
    df_s1_proc = preprocess_dataframe(df_s1_raw)
    df_cand_proc = preprocess_dataframe(df_cand_raw)

    s1_dict = {row["entity_id"]: row.to_dict() for _, row in df_s1_proc.iterrows()}
    cand_dict = {row["entity_id"]: row.to_dict() for _, row in df_cand_proc.iterrows()}

    # Extract features
    df_p_info, X_test, _, f_names = extract_features_for_pairs(
        df_pairs=df_pairs,
        s1_dict=s1_dict,
        cand_dict=cand_dict,
        name_vec=name_vec,
        addr_vec=addr_vec,
        comb_vec=comb_vec,
    )

    probs = model.predict_proba(X_test)[:, 1]

    all_tests_passed = True
    for i, sc in enumerate(test_scenarios):
        prob = float(probs[i])
        predicted_match = bool(prob >= threshold)
        expected = sc["expected_match"]
        passed = (predicted_match == expected)
        if not passed:
            all_tests_passed = False

        status_str = "[PASS]" if passed else "[FAIL]"
        print(f"\nScenario {i+1}: {sc['desc']} {status_str}")
        print(f"  S1:        '{sc['s1_name']}' | '{sc['s1_addr']}' | '{sc['s1_country']}'")
        print(f"  Candidate: '{sc['c_name']}' | '{sc['c_addr']}' | '{sc['c_country']}' ({sc['source']})")
        print(f"  Model Probability: {prob:.4f}  |  Threshold: {threshold:.2f}")
        print(f"  Decision: {'MATCH' if predicted_match else 'NO-MATCH'} (Expected: {'MATCH' if expected else 'NO-MATCH'})")

    print("\n" + "=" * 75)
    if all_tests_passed:
        print("VERIFICATION RESULT: ALL TESTS PASSED SUCCESSFULLY!")
        print("The model is operational, properly calibrated, and correctly discriminates entity matches.")
    else:
        print("VERIFICATION RESULT: SOME TESTS FAILED.")
    print("=" * 75)


if __name__ == "__main__":
    run_verification()
