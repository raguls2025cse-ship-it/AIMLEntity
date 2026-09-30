"""
train_model.py
Trains and compares binary classification models for entity resolution:
- LightGBM (tuned)
- HistGradientBoosting
- RandomForest
- LogisticRegression

Performs entity-level train/validation split (preventing leakage), sweeps decision thresholds
on entity-level Macro F0.5 (with proper singleton handling), and selects the winning model.
"""

import os
import json
import joblib
from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
try:
    from lightgbm import LGBMClassifier
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

from evaluate import evaluate_entity_macro_f05, sweep_thresholds


def get_models(random_seed: int = 42) -> Dict[str, Any]:
    """Returns candidate classification models with tuned parameters."""
    models = {
        "LogisticRegression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, C=1.0, random_state=random_seed))
        ]),
        "RandomForest": RandomForestClassifier(
            n_estimators=150,
            max_depth=16,
            min_samples_split=4,
            class_weight="balanced_subsample",
            random_state=random_seed,
            n_jobs=-1
        ),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            max_iter=150,
            max_depth=9,
            learning_rate=0.07,
            random_state=random_seed
        ),
    }

    if LIGHTGBM_AVAILABLE:
        models["LightGBM"] = LGBMClassifier(
            n_estimators=200,
            max_depth=8,
            num_leaves=63,
            learning_rate=0.06,
            subsample=0.85,
            colsample_bytree=0.85,
            class_weight="balanced",
            random_state=random_seed,
            verbose=-1
        )

    return models


def train_and_evaluate_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    val_pairs_df: pd.DataFrame,
    val_s1_ids: List[str],
    gt_matches_dict: Dict[str, Set[str]],
    feature_names: List[str],
    random_seed: int = 42,
) -> Tuple[str, Any, float, Dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """
    Trains all candidate models, sweeps threshold on validation set for each,
    and selects the winning model by entity-level Macro F0.5.
    """
    models = get_models(random_seed=random_seed)
    
    print("=" * 60)
    print("TRAINING AND COMPARING CLASSIFICATION MODELS:")
    print(f"Training pairs: {len(X_train):,} (Pos: {(y_train==1).sum():,}, Neg: {(y_train==0).sum():,})")
    print(f"Validation pairs: {len(X_val):,} (Pos: {(y_val==1).sum():,}, Neg: {(y_val==0).sum():,})")
    print(f"Validation S1 entities: {len(val_s1_ids):,}")
    print("=" * 60)

    comparison_results = []
    all_threshold_tables = []
    trained_models = {}
    model_val_probs = {}

    for name, model in models.items():
        print(f"\n--- Training {name} ---", flush=True)
        model.fit(X_train, y_train)
        trained_models[name] = model

        # Predict validation probabilities
        val_probs = model.predict_proba(X_val)[:, 1]
        model_val_probs[name] = val_probs

        # Sweep thresholds
        thresh_df, best_th, best_m = sweep_thresholds(
            val_s1_ids=val_s1_ids,
            gt_matches_dict=gt_matches_dict,
            val_pairs_df=val_pairs_df,
            val_probs=val_probs,
        )
        thresh_df["model_name"] = name
        all_threshold_tables.append(thresh_df)

        print(f"[{name}] Best Threshold: {best_th:.2f} | Macro F0.5: {best_m['macro_f05']:.4f} | Macro Precision: {best_m['macro_precision']:.4f} | Macro Recall: {best_m['macro_recall']:.4f}", flush=True)
        print(f"      Micro Precision: {best_m['micro_precision']:.4f} | Micro Recall: {best_m['micro_recall']:.4f} | TP: {best_m['total_tp']:,} | FP: {best_m['total_fp']:,} | FN: {best_m['total_fn']:,}", flush=True)

        comparison_results.append({
            "model_name": name,
            "best_threshold": best_th,
            "macro_f05": best_m["macro_f05"],
            "macro_precision": best_m["macro_precision"],
            "macro_recall": best_m["macro_recall"],
            "micro_f05": best_m["micro_f05"],
            "micro_precision": best_m["micro_precision"],
            "micro_recall": best_m["micro_recall"],
            "total_tp": best_m["total_tp"],
            "total_fp": best_m["total_fp"],
            "total_fn": best_m["total_fn"],
        })

    df_comparison = pd.DataFrame(comparison_results).sort_values("macro_f05", ascending=False)
    df_all_thresholds = pd.concat(all_threshold_tables, ignore_index=True)

    best_model_name = df_comparison.iloc[0]["model_name"]
    best_threshold = float(df_comparison.iloc[0]["best_threshold"])
    best_model = trained_models[best_model_name]
    best_val_probs = model_val_probs[best_model_name]

    # Re-compute full metrics for best model
    pred_mask = best_val_probs >= best_threshold
    s1_ids = val_pairs_df["source1_entity_id"].values
    cand_ids = val_pairs_df["candidate_entity_id"].values
    pred_pairs_dict = {}
    for s1, c in zip(s1_ids[pred_mask], cand_ids[pred_mask]):
        if s1 not in pred_pairs_dict:
            pred_pairs_dict[s1] = set()
        pred_pairs_dict[s1].add(c)

    best_metrics = evaluate_entity_macro_f05(val_s1_ids, gt_matches_dict, pred_pairs_dict)
    best_metrics["model_name"] = best_model_name
    best_metrics["best_threshold"] = best_threshold

    print("\n" + "=" * 60, flush=True)
    print(f"WINNING MODEL: {best_model_name} (Threshold = {best_threshold:.2f})", flush=True)
    print(f"Validation Macro F0.5: {best_metrics['macro_f05']:.4f}", flush=True)
    print(f"Validation Macro Precision: {best_metrics['macro_precision']:.4f}", flush=True)
    print(f"Validation Macro Recall: {best_metrics['macro_recall']:.4f}", flush=True)
    print("=" * 60 + "\n", flush=True)

    return best_model_name, best_model, best_threshold, best_metrics, df_comparison, df_all_thresholds
