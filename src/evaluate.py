"""
evaluate.py
Entity-level Macro F0.5 evaluation and threshold optimization.
Includes singleton handling and macro averaging per Source-1 entity.
"""

from collections import defaultdict
from typing import Dict, List, Set, Tuple, Any
import numpy as np
import pandas as pd


def compute_f_beta(precision: float, recall: float, beta: float = 0.5) -> float:
    """Computes F_beta score: (1 + beta^2) * P * R / (beta^2 * P + R). For beta=0.5: 1.25 * P * R / (0.25 * P + R)."""
    if precision <= 0.0 or recall <= 0.0:
        return 0.0
    beta_sq = beta ** 2
    denominator = (beta_sq * precision) + recall
    if denominator == 0.0:
        return 0.0
    return ((1.0 + beta_sq) * precision * recall) / denominator


def evaluate_entity_macro_f05(
    val_s1_ids: List[str],
    gt_matches_dict: Dict[str, Set[str]],
    pred_pairs_dict: Dict[str, Set[str]],
) -> Dict[str, Any]:
    """
    Evaluates Macro F0.5, Precision, and Recall across all validation Source-1 entities.
    Properly handles singleton entities:
    - If True = 0 and Pred = 0 -> P=1.0, R=1.0, F0.5=1.0
    - If True = 0 and Pred > 0 -> P=0.0, R=0.0, F0.5=0.0
    - If True > 0 and Pred = 0 -> P=0.0, R=0.0, F0.5=0.0
    - If True > 0 and Pred > 0 -> standard P, R, F0.5 calculation
    """
    n_entities = len(val_s1_ids)
    if n_entities == 0:
        return {}

    entity_precisions = []
    entity_recalls = []
    entity_f05s = []

    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_true_matches = 0
    total_pred_matches = 0
    singleton_count = 0
    singleton_correct = 0

    for s1_id in val_s1_ids:
        true_set = gt_matches_dict.get(s1_id, set())
        pred_set = pred_pairs_dict.get(s1_id, set())

        num_true = len(true_set)
        num_pred = len(pred_set)

        total_true_matches += num_true
        total_pred_matches += num_pred

        if num_true == 0:
            singleton_count += 1
            if num_pred == 0:
                # Correct singleton
                p, r, f05 = 1.0, 1.0, 1.0
                singleton_correct += 1
            else:
                # False positive on singleton
                p, r, f05 = 0.0, 0.0, 0.0
                total_fp += num_pred
        else:
            if num_pred == 0:
                p, r, f05 = 0.0, 0.0, 0.0
                total_fn += num_true
            else:
                tp = len(true_set.intersection(pred_set))
                fp = num_pred - tp
                fn = num_true - tp

                total_tp += tp
                total_fp += fp
                total_fn += fn

                p = tp / num_pred if num_pred > 0 else 0.0
                r = tp / num_true if num_true > 0 else 0.0
                f05 = compute_f_beta(p, r, beta=0.5)

        entity_precisions.append(p)
        entity_recalls.append(r)
        entity_f05s.append(f05)

    macro_precision = float(np.mean(entity_precisions))
    macro_recall = float(np.mean(entity_recalls))
    macro_f05 = float(np.mean(entity_f05s))

    # Micro metrics
    micro_precision = total_tp / total_pred_matches if total_pred_matches > 0 else 0.0
    micro_recall = total_tp / total_true_matches if total_true_matches > 0 else 0.0
    micro_f05 = compute_f_beta(micro_precision, micro_recall, beta=0.5)

    return {
        "macro_f05": macro_f05,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "micro_f05": micro_f05,
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "total_true_matches": total_true_matches,
        "total_predicted_matches": total_pred_matches,
        "total_tp": total_tp,
        "total_fp": total_fp,
        "total_fn": total_fn,
        "total_s1_entities": n_entities,
        "num_singletons": singleton_count,
        "singleton_correct": singleton_correct,
    }


def sweep_thresholds(
    val_s1_ids: List[str],
    gt_matches_dict: Dict[str, Set[str]],
    val_pairs_df: pd.DataFrame,
    val_probs: np.ndarray,
    thresholds: List[float] = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95],
) -> Tuple[pd.DataFrame, float, Dict[str, Any]]:
    """
    Sweeps decision probability thresholds on the validation set
    and returns the threshold results dataframe, optimal threshold, and best metrics.
    """
    results = []
    best_threshold = 0.5
    best_macro_f05 = -1.0
    best_metrics = {}

    s1_ids = val_pairs_df["source1_entity_id"].values
    cand_ids = val_pairs_df["candidate_entity_id"].values

    for thresh in thresholds:
        pred_mask = val_probs >= thresh
        pred_pairs_dict = defaultdict(set)
        
        for s1, c in zip(s1_ids[pred_mask], cand_ids[pred_mask]):
            pred_pairs_dict[s1].add(c)

        metrics = evaluate_entity_macro_f05(val_s1_ids, gt_matches_dict, pred_pairs_dict)
        metrics["threshold"] = thresh
        results.append(metrics)

        if metrics["macro_f05"] > best_macro_f05:
            best_macro_f05 = metrics["macro_f05"]
            best_threshold = thresh
            best_metrics = metrics

    df_results = pd.DataFrame(results)
    return df_results, best_threshold, best_metrics
