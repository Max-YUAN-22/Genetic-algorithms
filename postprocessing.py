#!/usr/bin/env python3
"""
Postprocessing utilities for medical image segmentation (brain tumor).

Functions here operate on discrete label maps (H, W) or (D, H, W) and are
designed to improve WT/TC/ET metrics by removing noise, filling small holes,
and enforcing simple anatomical plausibility constraints.
"""

from typing import Optional, Dict, Any, Tuple
import numpy as np

try:
    from scipy.ndimage import label as cc_label, binary_fill_holes, binary_opening, binary_closing
    from scipy.ndimage import generate_binary_structure
except Exception:
    cc_label = None
    binary_fill_holes = None
    binary_opening = None
    binary_closing = None
    generate_binary_structure = None


class PostprocessConfig:
    """Configuration for segmentation postprocessing."""

    def __init__(
        self,
        min_component_size: int = 50,
        fill_holes: bool = True,
        opening_radius: int = 0,
        closing_radius: int = 1,
        keep_largest_per_class: bool = False,
        apply_to_regions: Tuple[int, ...] = (1, 2, 3),
        is_3d: bool = False,
    ) -> None:
        self.min_component_size = min_component_size
        self.fill_holes = fill_holes
        self.opening_radius = opening_radius
        self.closing_radius = closing_radius
        self.keep_largest_per_class = keep_largest_per_class
        self.apply_to_regions = apply_to_regions
        self.is_3d = is_3d


def _get_structure(is_3d: bool, connectivity: int = 1):
    if generate_binary_structure is None:
        return None
    dims = 3 if is_3d else 2
    return generate_binary_structure(dims, connectivity)


def _morph(binary: np.ndarray, config: PostprocessConfig) -> np.ndarray:
    struct = _get_structure(config.is_3d, connectivity=1)

    processed = binary.copy()

    # Small opening to remove salt noise
    if binary_opening is not None and config.opening_radius > 0:
        processed = binary_opening(processed, structure=struct, iterations=config.opening_radius)

    # Closing to fill small gaps
    if binary_closing is not None and config.closing_radius > 0:
        processed = binary_closing(processed, structure=struct, iterations=config.closing_radius)

    # Optional hole filling
    if binary_fill_holes is not None and config.fill_holes:
        processed = binary_fill_holes(processed)

    # Remove tiny components
    if cc_label is not None and config.min_component_size > 0:
        labeled, num = cc_label(processed, structure=struct)
        if num > 0:
            counts = np.bincount(labeled.ravel())
            keep = np.zeros_like(counts, dtype=bool)
            keep[0] = False
            keep[np.where(counts >= config.min_component_size)] = True
            processed = keep[labeled]

    return processed.astype(np.uint8)


def _keep_largest_component(binary: np.ndarray, is_3d: bool) -> np.ndarray:
    if cc_label is None:
        return binary
    struct = _get_structure(is_3d, connectivity=1)
    labeled, num = cc_label(binary, structure=struct)
    if num <= 1:
        return binary
    counts = np.bincount(labeled.ravel())
    counts[0] = 0
    largest = counts.argmax()
    return (labeled == largest).astype(np.uint8)


def postprocess_segmentation(pred: np.ndarray, config: Optional[PostprocessConfig] = None) -> np.ndarray:
    """
    Postprocess a discrete segmentation map.

    Args:
        pred: ndarray of shape (H, W) or (D, H, W) with class indices {0..C-1}
        config: PostprocessConfig

    Returns:
        ndarray with the same shape and dtype uint8 of postprocessed labels
    """
    if config is None:
        config = PostprocessConfig()

    is_3d = config.is_3d or (pred.ndim == 3)
    out = pred.copy()

    for cls in config.apply_to_regions:
        binary = (out == cls).astype(np.uint8)
        binary = _morph(binary, config)
        if config.keep_largest_per_class:
            binary = _keep_largest_component(binary, is_3d)
        out[out == cls] = 0
        out[binary.astype(bool)] = cls

    return out.astype(np.uint8)


def postprocess_wt_tc_et(pred: np.ndarray, config: Optional[PostprocessConfig] = None) -> np.ndarray:
    """
    Focused postprocessing tailored for BraTS WT/TC/ET targets.
    Applies stronger filtering on ET to stabilize HD95 and removes scattered noise from WT/TC.
    """
    if config is None:
        config = PostprocessConfig()

    # Copy and process by class with class-specific min sizes
    out = pred.copy()

    # Whole Tumor (WT) uses union of all tumor classes; here we treat per-class
    class_specific_min = {1: max(30, config.min_component_size), 2: max(50, config.min_component_size), 3: max(20, config.min_component_size)}

    for cls, min_size in class_specific_min.items():
        local_cfg = PostprocessConfig(
            min_component_size=min_size,
            fill_holes=config.fill_holes,
            opening_radius=config.opening_radius,
            closing_radius=config.closing_radius,
            keep_largest_per_class=config.keep_largest_per_class,
            apply_to_regions=(cls,),
            is_3d=config.is_3d or (pred.ndim == 3),
        )
        binary = (out == cls).astype(np.uint8)
        binary = _morph(binary, local_cfg)
        if local_cfg.keep_largest_per_class:
            binary = _keep_largest_component(binary, local_cfg.is_3d)
        out[out == cls] = 0
        out[binary.astype(bool)] = cls

    return out.astype(np.uint8)



