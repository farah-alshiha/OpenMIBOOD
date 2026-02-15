#!/usr/bin/env python3
import argparse
from pathlib import Path

import pandas as pd
from tqdm import tqdm


def normalize_id(s: str) -> str:
    return str(s).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_root", required=True, help=".../raw/atlas_r2")
    ap.add_argument("--out_root", required=True, help=".../processed/atlas_r2")
    ap.add_argument("--metadata_xlsx", required=True, help=".../raw/atlas_r2/metadata.xlsx")
    ap.add_argument("--metadata_sheet", default=None, help="Optional sheet name; default reads first sheet")
    args = ap.parse_args()

    raw_root = Path(args.raw_root)
    train_root = raw_root / "Training"
    if not train_root.exists():
        raise FileNotFoundError(f"Expected {train_root} to exist.")

    out_root = Path(args.out_root)
    t1_out = out_root / "t1"
    mask_out = out_root / "lesion_masks"
    t1_out.mkdir(parents=True, exist_ok=True)
    mask_out.mkdir(parents=True, exist_ok=True)

    # ---- Load metadata.xlsx ----
    meta_xlsx = Path(args.metadata_xlsx)
    if not meta_xlsx.exists():
        raise FileNotFoundError(f"Metadata file not found: {meta_xlsx}")

    meta_df = pd.read_excel(meta_xlsx, sheet_name=args.metadata_sheet) if args.metadata_sheet else pd.read_excel(meta_xlsx)

    cols = {c.lower().strip(): c for c in meta_df.columns}

    def find_col(cands):
        for c in cands:
            if c in cols:
                return cols[c]
        return None

    subj_col = find_col(["subject id", "subject_id", "subject"])
    sess_col = find_col(["session", "ses", "session id"])
    if subj_col is None or sess_col is None:
        raise ValueError(f"Could not find Subject/Session columns. Found: {list(meta_df.columns)}")

    meta_df = meta_df.copy()
    meta_df["subject_id_key"] = meta_df[subj_col].apply(normalize_id)
    meta_df["session_key"] = meta_df[sess_col].apply(normalize_id)

    # ---- Find all T1 and lesion masks in Training ----
    t1_files = list(train_root.rglob("*_T1w.nii.gz"))

    # ATLAS lesion masks look like: *_desc-T1lesion_mask.nii.gz
    lesion_masks = list(train_root.rglob("*_desc-T1lesion_mask.nii.gz"))

    print(f"Found T1 files: {len(t1_files)}")
    print(f"Found lesion mask files: {len(lesion_masks)}")

    # Index lesion masks by a shared stem:
    # e.g., sub-r001s001_ses-1_space-..._label-L -> (remove suffix _desc-T1lesion_mask.nii.gz)
    mask_map = {}
    for m in lesion_masks:
        key = m.name.replace("_desc-T1lesion_mask.nii.gz", "")
        # strip any _label-... chunk (e.g., _label-L)
        key = key.split("_label-")[0]
        mask_map[key] = m


    rows = []
    for t1 in tqdm(sorted(t1_files), desc="Processing ATLAS Training"):
        key = t1.name.replace("_T1w.nii.gz", "")  # same prefix used for mask_map
        if key not in mask_map:
            # Sometimes naming differs; print a few misses for debugging
            # but don't spam too much
            continue

        m = mask_map[key]

        # Parse subject/session from the beginning of filename
        # sub-r001s001_ses-1_...
        parts = key.split("_")
        subj_id = parts[0]  # sub-r001s001
        session = parts[1] if len(parts) > 1 else ""  # ses-1

        out_t1 = t1_out / f"{key}_T1w.nii.gz"
        out_mk = mask_out / f"{key}_lesion.nii.gz"

        out_t1.write_bytes(t1.read_bytes())
        out_mk.write_bytes(m.read_bytes())

        rows.append({
            "subject_id": subj_id,
            "session": session,
            "subject_ses": f"{subj_id}_{session}",
            "key_prefix": key,
            "t1_path": str(out_t1),
            "lesion_mask_path": str(out_mk),
            "raw_t1": str(t1),
            "raw_mask": str(m),
        })

    files_df = pd.DataFrame(rows)
    if files_df.empty:
        # Show an example key to help debug instantly
        example_t1 = t1_files[0].name if t1_files else None
        example_mask = lesion_masks[0].name if lesion_masks else None
        raise RuntimeError(
            "No ATLAS T1/mask pairs were found after pairing.\n"
            f"Example T1: {example_t1}\nExample mask: {example_mask}\n"
            "Likely the pairing key differs; share one full T1 filename and one full mask filename."
        )

    # ---- Merge with metadata ----
    files_df["subject_id_key"] = files_df["subject_id"].apply(normalize_id)
    files_df["session_key"] = files_df["session"].apply(normalize_id)

    merged = files_df.merge(
        meta_df,
        how="left",
        left_on=["subject_id_key", "session_key"],
        right_on=["subject_id_key", "session_key"],
        suffixes=("", "_meta"),
    )

    out_csv = out_root / "meta.csv"
    merged.to_csv(out_csv, index=False)

    print(f"\nSaved paired subjects: {len(merged)}")
    print(f"Saved meta.csv: {out_csv}")
    print("Merge missing rows (no metadata match):", int(merged[subj_col].isna().sum()) if subj_col in merged.columns else "n/a")


if __name__ == "__main__":
    main()
