#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta_csv", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    np.random.seed(args.seed)

    meta = pd.read_csv(args.meta_csv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------
    # RANDOM SUBJECT SPLIT
    # -------------------------
    subjects = meta["subject_id"].unique()
    np.random.shuffle(subjects)

    n = len(subjects)
    n_train = int(0.7 * n)
    n_val = int(0.15 * n)

    train_subj = subjects[:n_train]
    val_subj = subjects[n_train:n_train+n_val]
    test_subj = subjects[n_train+n_val:]

    meta_random = meta.copy()
    meta_random["split"] = "train"
    meta_random.loc[meta_random["subject_id"].isin(val_subj), "split"] = "val"
    meta_random.loc[meta_random["subject_id"].isin(test_subj), "split"] = "test"

    meta_random[["subject_ses", "split"]].to_csv(
        out_dir / "random_split.csv", index=False
    )

    print("Random split created:")
    print(meta_random["split"].value_counts())

    # -------------------------
    # SCANNER SHIFT SPLIT
    # -------------------------
    # Try to detect scanner manufacturer column automatically
    scanner_cols = [c for c in meta.columns if "scanner" in c.lower() or "manufacturer" in c.lower()]
    if len(scanner_cols) == 0:
        print("No scanner column detected. Skipping scanner split.")
        return

    scanner_col = scanner_cols[0]
    print(f"Using scanner column: {scanner_col}")

    scanners = meta[scanner_col].dropna().unique()
    if len(scanners) < 2:
        print("Not enough scanner diversity. Skipping scanner split.")
        return

    # Pick first scanner for training, second for test (simple baseline)
    train_scanner = scanners[0]
    test_scanner = scanners[1]

    meta_scanner = meta.copy()
    meta_scanner["split"] = "train"
    meta_scanner.loc[meta_scanner[scanner_col] == test_scanner, "split"] = "test"

    meta_scanner[["subject_ses", "split"]].to_csv(
        out_dir / "scanner_split.csv", index=False
    )

    print("Scanner split created:")
    print(meta_scanner["split"].value_counts())


if __name__ == "__main__":
    main()
