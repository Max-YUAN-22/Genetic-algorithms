#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Convert binary tumor masks to YOLO segmentation polygon labels.

Inputs:
  - --mask_dir: directory of binary masks (PNG/JPG), filename stem must match image stem
  - --img_dir:  directory of corresponding images (to get width/height for normalization)
  - --out_labels: output directory for YOLO .txt files

Notes:
  - Each connected component yields one YOLO row: <class> x1 y1 x2 y2 ... xn yn (normalized to [0,1])
  - If a contour has too few points, it will be skipped
  - Default class_id=0 (tumor)

Example:
  python examples/mask_to_yolo_poly.py \
    --mask_dir /data/ct_mri_fused/train/labels \
    --img_dir /data/ct_mri_fused/train/images \
    --out_labels /data/ct_mri_fused_yolo/train/labels \
    --class_id 0 --min_points 6 --approx_epsilon 0.002
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np


def find_mask_file(stem: str, mask_dir: Path) -> Path | None:
    for ext in (".png", ".jpg", ".jpeg", ".tif", ".tiff"):
        p = mask_dir / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def load_image_size(img_dir: Path, stem: str) -> Tuple[int, int]:
    for ext in (".png", ".jpg", ".jpeg", ".tif", ".tiff"):
        p = img_dir / f"{stem}{ext}"
        if p.exists():
            im = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
            if im is None:
                break
            h, w = im.shape[:2]
            return w, h
    raise FileNotFoundError(f"Image for stem '{stem}' not found in {img_dir}")


def mask_to_polygons(mask: np.ndarray, approx_epsilon: float = 0.002, min_points: int = 6) -> list[np.ndarray]:
    # Ensure binary
    _, bin_ = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(bin_, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    polys: list[np.ndarray] = []
    for cnt in contours:
        if cnt.shape[0] < min_points:
            continue
        peri = cv2.arcLength(cnt, True)
        eps = approx_epsilon * peri
        approx = cv2.approxPolyDP(cnt, eps, True)
        if approx.shape[0] >= min_points:
            polys.append(approx[:, 0, :])  # (N,2)
    return polys


def write_yolo_seg(label_path: Path, polys: list[np.ndarray], class_id: int, size_wh: Tuple[int, int]):
    w, h = size_wh
    lines = []
    for poly in polys:
        xs = poly[:, 0].astype(np.float32) / max(w, 1)
        ys = poly[:, 1].astype(np.float32) / max(h, 1)
        coords = np.stack([xs, ys], axis=1).reshape(-1)
        parts = [str(class_id)] + [f"{v:.6f}" for v in coords.tolist()]
        lines.append(" ".join(parts))
    if lines:
        label_path.parent.mkdir(parents=True, exist_ok=True)
        label_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description="Convert binary masks to YOLO polygon labels")
    ap.add_argument("--mask_dir", type=str, required=True, help="Directory with binary masks (PNG/JPG)")
    ap.add_argument("--img_dir", type=str, required=True, help="Directory with paired images (to read width/height)")
    ap.add_argument("--out_labels", type=str, required=True, help="Output labels directory")
    ap.add_argument("--class_id", type=int, default=0, help="YOLO class id for tumor")
    ap.add_argument("--approx_epsilon", type=float, default=0.002, help="Approx epsilon factor relative to perimeter")
    ap.add_argument("--min_points", type=int, default=6, help="Minimum polygon points to keep a contour")
    args = ap.parse_args()

    mask_dir = Path(args.mask_dir)
    img_dir = Path(args.img_dir)
    out_labels = Path(args.out_labels)

    # Iterate by image stems in img_dir
    stems = sorted({p.stem for p in img_dir.glob("*.*")})
    converted = 0
    for stem in stems:
        mfile = find_mask_file(stem, mask_dir)
        if mfile is None:
            continue
        mask = cv2.imread(str(mfile), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        size_wh = load_image_size(img_dir, stem)
        polys = mask_to_polygons(mask, approx_epsilon=args.approx_epsilon, min_points=args.min_points)
        if not polys:
            continue
        label_path = out_labels / f"{stem}.txt"
        write_yolo_seg(label_path, polys, class_id=args.class_id, size_wh=size_wh)
        converted += 1

    print(f"Done. Converted {converted} masks to YOLO polygon labels at: {out_labels}")


if __name__ == "__main__":
    main()




