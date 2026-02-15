#!/usr/bin/env python3
import argparse
import os
import re
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from tqdm import tqdm


def find_first(patterns, root: Path):
    """Return first file matching any glob pattern under root."""
    for pat in patterns:
        hits = list(root.glob(pat))
        if hits:
            return hits[0]
    return None


def extract_subject_id_from_filename(fname: str) -> str:
    """
    Example: 900-00-5299_2005.03.22_t1_LPS_rSRI.nii.gz
    We'll use the prefix before first underscore: 900-00-5299
    """
    base = Path(fname).name
    return base.split("_")[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_root", required=True, help=".../raw/rembrandt (contains extracted/)")
    ap.add_argument("--out_root", required=True, help=".../processed/rembrandt")
    args = ap.parse_args()

    raw_root = Path(args.raw_root)
    extracted = raw_root / "extracted"
    out_root = Path(args.out_root)

    t1_out = out_root / "t1"
    mask_out = out_root / "tumor_masks"
    t1_out.mkdir(parents=True, exist_ok=True)
    mask_out.mkdir(parents=True, exist_ok=True)

    if not extracted.exists():
        raise FileNotFoundError(f"Expected {extracted} to exist. Unzip into extracted/ first.")

    rows = []

    # Patient folders can be nested; we search for T1 files and then find labels nearby
    t1_files = list(extracted.rglob("*_t1_*rSRI.nii.gz"))
    if len(t1_files) == 0:
        # fallback: any file containing "_t1_" and ".nii.gz"
        t1_files = list(extracted.rglob("*_t1_*.nii.gz"))

    for t1_path in tqdm(sorted(t1_files), desc="Processing REMBRANDT"):
        subj = extract_subject_id_from_filename(t1_path.name)
        parent = t1_path.parent

        # labels file often ends with GlistrBoost_out-labels.nii (not gz)
        labels = find_first(
            patterns=["*_GlistrBoost_out-labels.nii", "*_GlistrBoost_out-labels.nii.gz"],
            root=parent
        )
        if labels is None:
            # search a bit higher if structure differs
            labels = find_first(
                patterns=["*_GlistrBoost_out-labels.nii", "*_GlistrBoost_out-labels.nii.gz"],
                root=parent.parent
            )

        if labels is None:
            print(f"[WARN] No labels found for {t1_path}. Skipping.")
            continue

        # Load and create binary tumor mask: labels > 0
        lab_img = nib.load(str(labels))
        lab = lab_img.get_fdata()
        tumor = (lab > 0).astype(np.uint8)

        # Save standardized outputs
        out_t1 = t1_out / f"sub-remb-{subj}_T1w.nii.gz"
        out_mk = mask_out / f"sub-remb-{subj}_tumor.nii.gz"

        # Copy T1 by saving loaded image (keeps affine/header)
        t1_img = nib.load(str(t1_path))
        nib.save(t1_img, str(out_t1))

        tumor_img = nib.Nifti1Image(tumor, affine=lab_img.affine, header=lab_img.header)
        nib.save(tumor_img, str(out_mk))

        rows.append({
            "subject_id": f"sub-remb-{subj}",
            "t1_path": str(out_t1),
            "tumor_mask_path": str(out_mk),
            "raw_t1": str(t1_path),
            "raw_labels": str(labels),
        })

    meta = pd.DataFrame(rows).sort_values("subject_id")
    meta_path = out_root / "meta.csv"
    meta.to_csv(meta_path, index=False)
    print(f"\nSaved {len(meta)} subjects to {out_root}")
    print(f"Meta: {meta_path}")


if __name__ == "__main__":
    main()
