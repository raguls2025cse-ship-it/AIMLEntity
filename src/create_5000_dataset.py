"""
create_5000_dataset.py
Memory-efficient extraction of 5,000 Source-1 entities, all their true matches
from Source-2 and Source-3 via Ground Truth, plus negative samples for ML training.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd


def create_development_dataset(
    train_dir: str = "train",
    output_dir: str = "development/5000_dataset",
    n_s1: int = 5000,
    n_negative_s2: int = 5000,
    n_negative_s3: int = 5000,
    random_seed: int = 42,
    chunksize: int = 100000,
):
    os.makedirs(output_dir, exist_ok=True)
    rng = np.random.RandomState(random_seed)

    print("=" * 60)
    print("STEP 1: Counting rows in train_source1.tsv...")
    total_s1 = 0
    s1_path = os.path.join(train_dir, "train_source1.tsv")
    if not os.path.exists(s1_path):
        raise FileNotFoundError(f"Missing {s1_path}")

    for chunk in pd.read_csv(s1_path, sep="\t", usecols=["entity_id"], chunksize=chunksize):
        total_s1 += len(chunk)
    print(f"Total Source-1 records found: {total_s1:,}")

    # Random selection of indices
    if n_s1 > total_s1:
        raise ValueError(f"Requested {n_s1} records, but only {total_s1} available.")
    
    selected_indices = set(rng.choice(total_s1, size=n_s1, replace=False))
    print(f"Randomly selected {len(selected_indices):,} unique row indices (seed={random_seed}).")

    print("\nSTEP 2: Extracting exactly 5,000 Source-1 records in chunks...")
    s1_rows = []
    current_idx = 0
    for chunk in pd.read_csv(s1_path, sep="\t", chunksize=chunksize):
        chunk_indices = set(range(current_idx, current_idx + len(chunk)))
        overlap = chunk_indices.intersection(selected_indices)
        if overlap:
            local_indices = [idx - current_idx for idx in overlap]
            s1_rows.append(chunk.iloc[local_indices])
        current_idx += len(chunk)

    df_s1 = pd.concat(s1_rows, ignore_index=True)
    assert len(df_s1) == n_s1, f"Expected {n_s1} records, got {len(df_s1)}"
    
    s1_out_path = os.path.join(output_dir, "train_source1.tsv")
    df_s1.to_csv(s1_out_path, sep="\t", index=False)
    print(f"Saved 5,000 Source-1 records to {s1_out_path}")

    selected_s1_ids = set(df_s1["entity_id"].dropna().unique())

    print("\nSTEP 3: Extracting Ground Truth for selected Source-1 entities...")
    gt_path = os.path.join(train_dir, "train_ground_truth.tsv")
    gt_rows = []
    for chunk in pd.read_csv(gt_path, sep="\t", chunksize=chunksize):
        matched_chunk = chunk[chunk["source1_entity_id"].isin(selected_s1_ids)]
        if not matched_chunk.empty:
            gt_rows.append(matched_chunk)

    df_gt = pd.concat(gt_rows, ignore_index=True) if gt_rows else pd.DataFrame(columns=["source1_entity_id", "matched_entity_ids"])

    # Ensure all 5,000 S1 IDs are present in GT (if any missing, add as singletons)
    existing_gt_ids = set(df_gt["source1_entity_id"].unique())
    missing_s1_ids = selected_s1_ids - existing_gt_ids
    if missing_s1_ids:
        missing_df = pd.DataFrame({
            "source1_entity_id": list(missing_s1_ids),
            "matched_entity_ids": [""] * len(missing_s1_ids)
        })
        df_gt = pd.concat([df_gt, missing_df], ignore_index=True)

    gt_out_path = os.path.join(output_dir, "train_ground_truth.tsv")
    df_gt.to_csv(gt_out_path, sep="\t", index=False)
    print(f"Saved Ground Truth ({len(df_gt)} records) to {gt_out_path}")

    # Parse matched IDs for S2 and S3
    true_s2_ids = set()
    true_s3_ids = set()
    num_singletons = 0
    total_positive_pairs = 0

    for _, row in df_gt.iterrows():
        raw_matches = row["matched_entity_ids"]
        if pd.isna(raw_matches) or str(raw_matches).strip() == "":
            num_singletons += 1
            continue
        matches = [m.strip() for m in str(raw_matches).split(",") if m.strip()]
        if not matches:
            num_singletons += 1
            continue
        for m in matches:
            total_positive_pairs += 1
            if m.startswith("S2-"):
                true_s2_ids.add(m)
            elif m.startswith("S3-"):
                true_s3_ids.add(m)

    print(f"Ground Truth Summary for 5,000 S1 entities:")
    print(f"  - Total Positive Pairs: {total_positive_pairs}")
    print(f"  - Unique True S2 IDs: {len(true_s2_ids)}")
    print(f"  - Unique True S3 IDs: {len(true_s3_ids)}")
    print(f"  - Singleton S1 Entities (no match): {num_singletons}")

    print("\nSTEP 4: Extracting True & Sampled Negative S2 records from train_source2.tsv...")
    s2_path = os.path.join(train_dir, "train_source2.tsv")
    true_s2_rows = []
    sampled_neg_s2 = []
    neg_s2_prob = (n_negative_s2 * 1.5) / 2000000.0  # approximate sampling rate

    for chunk in pd.read_csv(s2_path, sep="\t", chunksize=chunksize):
        # True matches
        is_true = chunk["entity_id"].isin(true_s2_ids)
        if is_true.any():
            true_s2_rows.append(chunk[is_true])
        
        # Negative candidates
        non_matches = chunk[~is_true]
        if len(sampled_neg_s2) < n_negative_s2 and not non_matches.empty:
            sample_size = min(int(len(non_matches) * neg_s2_prob) + 20, len(non_matches))
            sampled_chunk = non_matches.sample(n=sample_size, random_state=rng.randint(0, 1000000))
            sampled_neg_s2.append(sampled_chunk)

    df_true_s2 = pd.concat(true_s2_rows, ignore_index=True) if true_s2_rows else pd.DataFrame(columns=["entity_id", "business_name", "business_address", "country"])
    df_neg_s2 = pd.concat(sampled_neg_s2, ignore_index=True) if sampled_neg_s2 else pd.DataFrame(columns=["entity_id", "business_name", "business_address", "country"])
    
    # Trim negative samples to exact target
    if len(df_neg_s2) > n_negative_s2:
        df_neg_s2 = df_neg_s2.sample(n=n_negative_s2, random_state=random_seed)

    df_s2 = pd.concat([df_true_s2, df_neg_s2], ignore_index=True).drop_duplicates(subset=["entity_id"])
    s2_out_path = os.path.join(output_dir, "train_source2.tsv")
    df_s2.to_csv(s2_out_path, sep="\t", index=False)
    print(f"Saved {len(df_s2)} Source-2 records ({len(df_true_s2)} true matches, {len(df_neg_s2)} negative samples) to {s2_out_path}")

    print("\nSTEP 5: Extracting True & Sampled Negative S3 records from train_source3.tsv...")
    s3_path = os.path.join(train_dir, "train_source3.tsv")
    true_s3_rows = []
    sampled_neg_s3 = []
    neg_s3_prob = (n_negative_s3 * 1.5) / 2000000.0

    for chunk in pd.read_csv(s3_path, sep="\t", chunksize=chunksize):
        # True matches
        is_true = chunk["entity_id"].isin(true_s3_ids)
        if is_true.any():
            true_s3_rows.append(chunk[is_true])
        
        # Negative candidates
        non_matches = chunk[~is_true]
        if len(sampled_neg_s3) < n_negative_s3 and not non_matches.empty:
            sample_size = min(int(len(non_matches) * neg_s3_prob) + 20, len(non_matches))
            sampled_chunk = non_matches.sample(n=sample_size, random_state=rng.randint(0, 1000000))
            sampled_neg_s3.append(sampled_chunk)

    df_true_s3 = pd.concat(true_s3_rows, ignore_index=True) if true_s3_rows else pd.DataFrame(columns=["entity_id", "business_name", "business_address", "country"])
    df_neg_s3 = pd.concat(sampled_neg_s3, ignore_index=True) if sampled_neg_s3 else pd.DataFrame(columns=["entity_id", "business_name", "business_address", "country"])

    if len(df_neg_s3) > n_negative_s3:
        df_neg_s3 = df_neg_s3.sample(n=n_negative_s3, random_state=random_seed)

    df_s3 = pd.concat([df_true_s3, df_neg_s3], ignore_index=True).drop_duplicates(subset=["entity_id"])
    s3_out_path = os.path.join(output_dir, "train_source3.tsv")
    df_s3.to_csv(s3_out_path, sep="\t", index=False)
    print(f"Saved {len(df_s3)} Source-3 records ({len(df_true_s3)} true matches, {len(df_neg_s3)} negative samples) to {s3_out_path}")

    print("\n" + "=" * 60)
    print("5,000 DEVELOPMENT DATASET CREATION COMPLETE!")
    print(f"  - Source 1 Count: {len(df_s1):,}")
    print(f"  - Source 2 Count: {len(df_s2):,}")
    print(f"  - Source 3 Count: {len(df_s3):,}")
    print(f"  - Ground Truth Records: {len(df_gt):,}")
    print(f"  - Total True Matches: {total_positive_pairs:,}")
    print(f"  - Singleton S1 Entities: {num_singletons:,}")
    print("=" * 60)

    return {
        "s1_count": len(df_s1),
        "s2_count": len(df_s2),
        "s3_count": len(df_s3),
        "gt_count": len(df_gt),
        "positive_pairs": total_positive_pairs,
        "singletons": num_singletons,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create 5,000 S1 Development Dataset")
    parser.add_argument("--train_dir", type=str, default="train")
    parser.add_argument("--output_dir", type=str, default="development/5000_dataset")
    parser.add_argument("--n_s1", type=int, default=5000)
    parser.add_argument("--n_neg_s2", type=int, default=5000)
    parser.add_argument("--n_neg_s3", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    create_development_dataset(
        train_dir=args.train_dir,
        output_dir=args.output_dir,
        n_s1=args.n_s1,
        n_negative_s2=args.n_neg_s2,
        n_negative_s3=args.n_neg_s3,
        random_seed=args.seed,
    )
