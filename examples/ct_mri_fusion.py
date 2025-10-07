#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CT/MRI Pairwise Registration and Fusion Utility

This script performs pairwise registration between CT (fixed) and MRI (moving) images,
warps MRI (and optional MRI-space masks) into CT space, and saves fused training images.

Fusion methods supported:
  - stack: stack [CT, MRI_aligned, average] into 3 channels (recommended)
  - avg:   weighted average 0.5*CT + 0.5*MRI_aligned (single-channel, auto-expand to 3)

Labels handling:
  - If you already have YOLO segmentation/detection labels in CT space, pass --labels_dir and
    they will be copied as-is since the reference space is unchanged (CT).
  - If you have binary mask images aligned to MRI space, pass --mri_masks_dir with
    --masks_source mri to warp them with the same transform into CT space and save.

Note: Converting binary masks to YOLO polygon labels is out of scope here. Use your existing
      CT-space YOLO labels or convert masks to polygons separately before training.

Example usage:
  python examples/ct_mri_fusion.py \
    --ct_dir /data/ct/train \
    --mri_dir /data/mri/train \
    --out_root /data/ct_mri_fused \
    --labels_dir /data/labels/train \
    --fusion_method stack --resize 640 640

  # With MRI-space binary masks to be warped into CT space:
  python examples/ct_mri_fusion.py \
    --ct_dir /data/ct/train --mri_dir /data/mri/train \
    --mri_masks_dir /data/mri_masks/train --masks_source mri \
    --out_root /data/ct_mri_fused --fusion_method stack
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Tuple, Optional

import cv2
import numpy as np


def read_image_grayscale(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    return img


def minmax_normalize(img: np.ndarray) -> np.ndarray:
    img = img.astype(np.float32)
    mn, mx = float(img.min()), float(img.max())
    if mx > mn:
        img = (img - mn) / (mx - mn)
    else:
        img = np.zeros_like(img, dtype=np.float32)
    return img


def clahe_enhance(img: np.ndarray, clip_limit: float = 2.0, tile_grid_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
    img_u8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(img_u8).astype(np.uint8)


def ecc_register_affine(
    moving: np.ndarray,
    fixed: np.ndarray,
    number_of_iterations: int = 300,
    termination_eps: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Register moving -> fixed using OpenCV ECC with affine model.
    Returns (warped_moving, 2x3 affine)
    """
    if moving.ndim != 2 or fixed.ndim != 2:
        raise ValueError("ECC registration expects single-channel images.")

    # ECC requires float32 in [0,1]
    mv = minmax_normalize(moving).astype(np.float32)
    fx = minmax_normalize(fixed).astype(np.float32)

    warp_matrix = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations, termination_eps)

    try:
        cc, warp_matrix = cv2.findTransformECC(
            fx,
            mv,
            warp_matrix,
            motionType=cv2.MOTION_AFFINE,
            criteria=criteria,
            inputMask=None,
            gaussFiltSize=5,
        )
        _ = cc  # not used further
    except cv2.error as e:
        raise RuntimeError(f"ECC registration failed: {e}")

    h, w = fixed.shape
    warped = cv2.warpAffine(moving, warp_matrix, (w, h), flags=cv2.INTER_LINEAR)
    return warped, warp_matrix


def warp_mask_affine(mask: np.ndarray, warp_matrix: np.ndarray, shape_wh: Tuple[int, int]) -> np.ndarray:
    w, h = shape_wh
    return cv2.warpAffine(mask, warp_matrix, (w, h), flags=cv2.INTER_NEAREST)


def to_three_channels(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.ndim == 3 and img.shape[2] == 3:
        return img
    raise ValueError("Unsupported image shape for 3-channel conversion.")


def fuse_images(
    ct_u8: np.ndarray,
    mri_aligned_u8: np.ndarray,
    method: str = "stack",
) -> np.ndarray:
    """Return fused 3-channel image."""
    method = method.lower()
    if method == "avg":
        avg = ((ct_u8.astype(np.float32) + mri_aligned_u8.astype(np.float32)) * 0.5).astype(np.uint8)
        return to_three_channels(avg)

    if method == "stack":
        ct_f = minmax_normalize(ct_u8)
        mri_f = minmax_normalize(mri_aligned_u8)
        avg = ((ct_f + mri_f) * 0.5)
        # Optionally enhance CT as first channel with CLAHE for contrast
        ct_enh = clahe_enhance(ct_f)
        avg_u8 = (np.clip(avg, 0, 1) * 255).astype(np.uint8)
        return np.dstack([ct_enh, mri_aligned_u8, avg_u8])

    raise ValueError(f"Unknown fusion method: {method}")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def pair_by_stem(ct_dir: Path, mri_dir: Path) -> list[Tuple[Path, Path]]:
    ct_map = {p.stem: p for p in sorted(ct_dir.glob("*")) if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".tif", ".tiff")}
    mri_map = {p.stem: p for p in sorted(mri_dir.glob("*")) if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".tif", ".tiff")}
    pairs = []
    for k, v in ct_map.items():
        if k in mri_map:
            pairs.append((v, mri_map[k]))
    return pairs


def main():
    parser = argparse.ArgumentParser(description="CT/MRI registration and fusion into CT space")
    parser.add_argument("--ct_dir", type=str, required=True, help="Directory of CT images (fixed/reference)")
    parser.add_argument("--mri_dir", type=str, required=True, help="Directory of MRI images (moving)")
    parser.add_argument("--out_root", type=str, required=True, help="Output root directory (will create images/ and optional labels/")
    parser.add_argument("--labels_dir", type=str, default="", help="Optional labels directory already in CT space (YOLO .txt). Will be copied by stem.")
    parser.add_argument("--mri_masks_dir", type=str, default="", help="Optional MRI-space binary masks to warp into CT space.")
    parser.add_argument("--masks_source", type=str, choices=["ct", "mri"], default="ct", help="Where masks/labels are defined. 'ct' means labels_dir is in CT space (no warp). 'mri' means mri_masks_dir will be warped.")
    parser.add_argument("--fusion_method", type=str, choices=["stack", "avg"], default="stack", help="Fusion method for output image")
    parser.add_argument("--resize", type=int, nargs=2, default=None, help="Optional resize (W H) applied to CT and MRI before registration")
    parser.add_argument("--suffix", type=str, default="", help="Optional suffix added to output image stems")

    args = parser.parse_args()

    ct_dir = Path(args.ct_dir)
    mri_dir = Path(args.mri_dir)
    out_root = Path(args.out_root)
    labels_dir: Optional[Path] = Path(args.labels_dir) if args.labels_dir else None
    mri_masks_dir: Optional[Path] = Path(args.mri_masks_dir) if args.mri_masks_dir else None

    out_images = out_root / "images"
    out_labels = out_root / "labels" if (labels_dir or mri_masks_dir) else None
    ensure_dir(out_images)
    if out_labels:
        ensure_dir(out_labels)

    pairs = pair_by_stem(ct_dir, mri_dir)
    if not pairs:
        raise RuntimeError("No CT/MRI pairs found by matching filename stems.")

    print(f"Found {len(pairs)} CT/MRI pairs. Processing...")

    warped_count = 0
    copied_labels = 0
    warped_masks = 0

    for ct_path, mri_path in pairs:
        stem = ct_path.stem
        ct = read_image_grayscale(ct_path)
        mri = read_image_grayscale(mri_path)

        if args.resize is not None:
            w, h = args.resize
            ct = cv2.resize(ct, (w, h), interpolation=cv2.INTER_AREA)
            mri = cv2.resize(mri, (w, h), interpolation=cv2.INTER_AREA)

        try:
            mri_aligned, warp_mat = ecc_register_affine(mri, ct)
            warped_count += 1
        except Exception as e:
            print(f"[WARN] Registration failed for pair {stem}: {e}. Skipping.")
            continue

        fused = fuse_images(ct, mri_aligned, method=args.fusion_method)
        out_name = f"{stem}{args.suffix}.png" if args.suffix else f"{stem}.png"
        cv2.imwrite(str(out_images / out_name), fused)

        # Handle labels or masks
        if out_labels is not None:
            if args.masks_source == "ct" and labels_dir is not None:
                # Copy YOLO .txt labels by stem
                src_label = labels_dir / f"{stem}.txt"
                if src_label.exists():
                    dst_label = out_labels / src_label.name
                    try:
                        data = src_label.read_text(encoding="utf-8")
                        dst_label.write_text(data, encoding="utf-8")
                        copied_labels += 1
                    except Exception as e:
                        print(f"[WARN] Failed to copy label for {stem}: {e}")
            elif args.masks_source == "mri" and mri_masks_dir is not None:
                # Warp MRI-space binary mask into CT space and save as PNG (not YOLO .txt)
                # Users should convert PNG mask to YOLO polygon if needed later.
                cand = None
                for ext in (".png", ".jpg", ".jpeg", ".tif", ".tiff"):
                    p = mri_masks_dir / f"{stem}{ext}"
                    if p.exists():
                        cand = p
                        break
                if cand is not None:
                    mask = read_image_grayscale(cand)
                    if args.resize is not None:
                        w, h = args.resize
                        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
                    warped = warp_mask_affine(mask, warp_mat, (ct.shape[1], ct.shape[0]))
                    # binarize
                    _, warped_bin = cv2.threshold(warped, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
                    out_mask_name = f"{stem}{args.suffix}_mask.png" if args.suffix else f"{stem}_mask.png"
                    cv2.imwrite(str(out_labels / out_mask_name), warped_bin)
                    warped_masks += 1

    print(f"Done. Saved fused images to: {out_images}")
    if out_labels is not None:
        print(f"Labels/Masks saved to: {out_labels}")
    print(f"Pairs processed: {len(pairs)}, registered: {warped_count}, labels copied: {copied_labels}, masks warped: {warped_masks}")


if __name__ == "__main__":
    main()




