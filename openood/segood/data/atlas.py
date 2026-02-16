from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List

import numpy as np
import pandas as pd
import nibabel as nib
import torch
from torch.utils.data import Dataset


def zscore_brain(vol: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    # Use non-zero voxels as crude brain mask (works ok after MNI space)
    mask = vol != 0
    if mask.sum() < 100:
        return (vol - vol.mean()) / (vol.std() + eps)
    mu = vol[mask].mean()
    sd = vol[mask].std()
    return (vol - mu) / (sd + eps)


@dataclass
class Atlas2DConfig:
    meta_csv: str
    split_csv: str
    split: str                 # "train" | "val" | "test"
    crop_hw: Tuple[int, int] = (256, 256)
    slice_axis: int = 2        # axial
    min_lesion_pixels: int = 10  # filter empty-ish slices


class AtlasLesion2DSlices(Dataset):
    """
    Returns:
      x: FloatTensor [1, H, W]
      y: LongTensor  [H, W]  (0/1)
      meta: dict
    """
    def __init__(self, cfg: Atlas2DConfig):
        self.cfg = cfg
        meta = pd.read_csv(cfg.meta_csv)
        split_df = pd.read_csv(cfg.split_csv)

        meta = meta.merge(split_df, on="subject_ses", how="inner")
        meta = meta[meta["split"] == cfg.split].reset_index(drop=True)
        if len(meta) == 0:
            raise ValueError(f"No rows found for split='{cfg.split}' in {cfg.split_csv}")

        self.rows = meta.to_dict("records")

        # Build slice index: list of (row_idx, slice_idx)
        self.index: List[Tuple[int, int]] = []
        for i, r in enumerate(self.rows):
            mpath = Path(r["lesion_mask_path"])
            m = nib.load(str(mpath)).get_fdata()
            m = (m > 0).astype(np.uint8)

            # choose slice axis
            n_slices = m.shape[cfg.slice_axis]
            for s in range(n_slices):
                sl = np.take(m, s, axis=cfg.slice_axis)
                if sl.sum() >= cfg.min_lesion_pixels or cfg.split != "train":
                    # keep all slices for val/test; filter mostly-empty for train
                    self.index.append((i, s))

        if len(self.index) == 0:
            raise RuntimeError("No slices indexed. Try lowering min_lesion_pixels.")

    def __len__(self):
        return len(self.index)

    def _center_crop(self, img: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        H, W = img.shape
        ch, cw = self.cfg.crop_hw
        ch = min(ch, H); cw = min(cw, W)
        y0 = (H - ch) // 2
        x0 = (W - cw) // 2
        return img[y0:y0+ch, x0:x0+cw], mask[y0:y0+ch, x0:x0+cw]

    def __getitem__(self, idx: int):
        row_idx, sidx = self.index[idx]
        r = self.rows[row_idx]

        img = nib.load(str(r["t1_path"])).get_fdata().astype(np.float32)
        msk = nib.load(str(r["lesion_mask_path"])).get_fdata()
        msk = (msk > 0).astype(np.uint8)

        img = zscore_brain(img)

        # extract slice
        img_sl = np.take(img, sidx, axis=self.cfg.slice_axis)
        msk_sl = np.take(msk, sidx, axis=self.cfg.slice_axis)

        # crop
        img_sl, msk_sl = self._center_crop(img_sl, msk_sl)

        x = torch.from_numpy(img_sl[None, ...])          # [1,H,W]
        y = torch.from_numpy(msk_sl.astype(np.int64))    # [H,W]

        meta = {
            "subject_id": r.get("subject_id", ""),
            "session": r.get("session", ""),
            "subject_ses": r.get("subject_ses", ""),
            "slice_idx": int(sidx),
        }
        return x, y, meta
