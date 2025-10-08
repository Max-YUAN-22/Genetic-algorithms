#!/usr/bin/env python3
"""
Medical Image Segmentation Metrics for Brain Tumor Analysis.

Comprehensive evaluation metrics specifically designed for medical image segmentation, including clinically relevant
measures for brain tumor assessment.

This module provides SCI Q2+ quality evaluation metrics that are essential for medical image analysis publications.
"""

from typing import Union

import numpy as np
from scipy import ndimage
from scipy.spatial.distance import directed_hausdorff


class MedicalSegmentationMetrics:
    """
    Comprehensive medical image segmentation metrics for brain tumor analysis.

    Implements clinically relevant metrics used in top-tier medical imaging journals.
    """

    def __init__(self, num_classes: int = 4, class_names: list[str] = None):
        """
        Initialize medical metrics calculator.

        Args:
            num_classes: Number of segmentation classes (default: 4 for background, core, edema, enhancing)
            class_names: Names of segmentation classes
        """
        self.num_classes = num_classes
        self.class_names = class_names or ["Background", "Core", "Edema", "Enhancing"]

    def dice_coefficient(self, pred: np.ndarray, target: np.ndarray, class_idx: int = None) -> Union[float, np.ndarray]:
        """
        Calculate Dice Similarity Coefficient (DSC).

        The most important metric for medical image segmentation.
        DSC = 2 * |A ∩ B| / (|A| + |B|)

        Args:
            pred: Predicted segmentation mask
            target: Ground truth segmentation mask
            class_idx: Specific class to calculate (None for all classes)

        Returns:
            Dice coefficient(s)
        """
        if class_idx is not None:
            pred_class = (pred == class_idx).astype(np.float32)
            target_class = (target == class_idx).astype(np.float32)

            intersection = np.sum(pred_class * target_class)
            total = np.sum(pred_class) + np.sum(target_class)

            return 2.0 * intersection / (total + 1e-8)

        dice_scores = []
        for i in range(1, self.num_classes):  # Skip background
            dice_scores.append(self.dice_coefficient(pred, target, i))

        return np.array(dice_scores)

    def jaccard_index(self, pred: np.ndarray, target: np.ndarray, class_idx: int = None) -> Union[float, np.ndarray]:
        """
        Calculate Jaccard Index (Intersection over Union).

        IoU = |A ∩ B| / |A ∪ B|

        Args:
            pred: Predicted segmentation mask
            target: Ground truth segmentation mask
            class_idx: Specific class to calculate (None for all classes)

        Returns:
            Jaccard index(es)
        """
        if class_idx is not None:
            pred_class = (pred == class_idx).astype(np.float32)
            target_class = (target == class_idx).astype(np.float32)

            intersection = np.sum(pred_class * target_class)
            union = np.sum(pred_class) + np.sum(target_class) - intersection

            return intersection / (union + 1e-8)

        jaccard_scores = []
        for i in range(1, self.num_classes):  # Skip background
            jaccard_scores.append(self.jaccard_index(pred, target, i))

        return np.array(jaccard_scores)

    def sensitivity_recall(
        self, pred: np.ndarray, target: np.ndarray, class_idx: int = None
    ) -> Union[float, np.ndarray]:
        """
        Calculate Sensitivity (Recall/True Positive Rate).

        Sensitivity = TP / (TP + FN)
        Critical for medical applications - measures ability to detect disease.

        Args:
            pred: Predicted segmentation mask
            target: Ground truth segmentation mask
            class_idx: Specific class to calculate (None for all classes)

        Returns:
            Sensitivity value(s)
        """
        if class_idx is not None:
            pred_class = (pred == class_idx).astype(np.float32)
            target_class = (target == class_idx).astype(np.float32)

            tp = np.sum(pred_class * target_class)
            fn = np.sum((1 - pred_class) * target_class)

            return tp / (tp + fn + 1e-8)

        sensitivity_scores = []
        for i in range(1, self.num_classes):  # Skip background
            sensitivity_scores.append(self.sensitivity_recall(pred, target, i))

        return np.array(sensitivity_scores)

    def specificity(self, pred: np.ndarray, target: np.ndarray, class_idx: int = None) -> Union[float, np.ndarray]:
        """
        Calculate Specificity (True Negative Rate).

        Specificity = TN / (TN + FP)
        Important for measuring false positive rate.

        Args:
            pred: Predicted segmentation mask
            target: Ground truth segmentation mask
            class_idx: Specific class to calculate (None for all classes)

        Returns:
            Specificity value(s)
        """
        if class_idx is not None:
            pred_class = (pred == class_idx).astype(np.float32)
            target_class = (target == class_idx).astype(np.float32)

            tn = np.sum((1 - pred_class) * (1 - target_class))
            fp = np.sum(pred_class * (1 - target_class))

            return tn / (tn + fp + 1e-8)

        specificity_scores = []
        for i in range(1, self.num_classes):  # Skip background
            specificity_scores.append(self.specificity(pred, target, i))

        return np.array(specificity_scores)

    def hausdorff_distance(
        self, pred: np.ndarray, target: np.ndarray, class_idx: int = None, percentile: int = 95
    ) -> Union[float, np.ndarray]:
        """
        Calculate Hausdorff Distance (HD95).

        Measures the maximum distance between surface points of segmentations.
        HD95 is more robust than HD100 as it ignores outliers.

        Args:
            pred: Predicted segmentation mask
            target: Ground truth segmentation mask
            class_idx: Specific class to calculate (None for all classes)
            percentile: Percentile for robust Hausdorff distance (default: 95)

        Returns:
            Hausdorff distance(s) in mm
        """

        def _hausdorff_distance_single(pred_mask: np.ndarray, target_mask: np.ndarray) -> float:
            if np.sum(pred_mask) == 0 or np.sum(target_mask) == 0:
                return float("inf")

            # Get surface points
            pred_surface = self._get_surface_points(pred_mask)
            target_surface = self._get_surface_points(target_mask)

            if len(pred_surface) == 0 or len(target_surface) == 0:
                return float("inf")

            # Calculate directed Hausdorff distances
            directed_hausdorff(pred_surface, target_surface)[0]
            directed_hausdorff(target_surface, pred_surface)[0]

            # Return 95th percentile
            distances = np.concatenate(
                [
                    np.min(np.linalg.norm(pred_surface[:, None] - target_surface, axis=2), axis=1),
                    np.min(np.linalg.norm(target_surface[:, None] - pred_surface, axis=2), axis=1),
                ]
            )

            return np.percentile(distances, percentile)

        if class_idx is not None:
            pred_class = (pred == class_idx).astype(np.uint8)
            target_class = (target == class_idx).astype(np.uint8)
            return _hausdorff_distance_single(pred_class, target_class)

        hd_scores = []
        for i in range(1, self.num_classes):  # Skip background
            pred_class = (pred == i).astype(np.uint8)
            target_class = (target == i).astype(np.uint8)
            hd_scores.append(_hausdorff_distance_single(pred_class, target_class))

        return np.array(hd_scores)

    def _get_surface_points(self, mask: np.ndarray) -> np.ndarray:
        """Extract surface points from binary mask."""
        # Get boundary using morphological operations
        structure = ndimage.generate_binary_structure(mask.ndim, 1)
        eroded = ndimage.binary_erosion(mask, structure)
        boundary = mask ^ eroded

        # Get coordinates of boundary points
        surface_points = np.column_stack(np.where(boundary))
        return surface_points

    def average_surface_distance(
        self, pred: np.ndarray, target: np.ndarray, class_idx: int = None
    ) -> Union[float, np.ndarray]:
        """
        Calculate Average Surface Distance (ASD).

        Average of distances from surface points of prediction to nearest surface points of target.

        Args:
            pred: Predicted segmentation mask
            target: Ground truth segmentation mask
            class_idx: Specific class to calculate (None for all classes)

        Returns:
            Average surface distance(s) in mm
        """

        def _asd_single(pred_mask: np.ndarray, target_mask: np.ndarray) -> float:
            if np.sum(pred_mask) == 0 or np.sum(target_mask) == 0:
                return float("inf")

            pred_surface = self._get_surface_points(pred_mask)
            target_surface = self._get_surface_points(target_mask)

            if len(pred_surface) == 0 or len(target_surface) == 0:
                return float("inf")

            # Calculate average surface distance
            distances1 = np.min(np.linalg.norm(pred_surface[:, None] - target_surface, axis=2), axis=1)
            distances2 = np.min(np.linalg.norm(target_surface[:, None] - pred_surface, axis=2), axis=1)

            return (np.mean(distances1) + np.mean(distances2)) / 2.0

        if class_idx is not None:
            pred_class = (pred == class_idx).astype(np.uint8)
            target_class = (target == class_idx).astype(np.uint8)
            return _asd_single(pred_class, target_class)

        asd_scores = []
        for i in range(1, self.num_classes):  # Skip background
            pred_class = (pred == i).astype(np.uint8)
            target_class = (target == i).astype(np.uint8)
            asd_scores.append(_asd_single(pred_class, target_class))

        return np.array(asd_scores)

    def volumetric_similarity(
        self, pred: np.ndarray, target: np.ndarray, class_idx: int = None
    ) -> Union[float, np.ndarray]:
        """
        Calculate Volumetric Similarity (VS).

        VS = 1 - |V_pred - V_target| / (V_pred + V_target)

        Args:
            pred: Predicted segmentation mask
            target: Ground truth segmentation mask
            class_idx: Specific class to calculate (None for all classes)

        Returns:
            Volumetric similarity value(s)
        """
        if class_idx is not None:
            pred_vol = np.sum(pred == class_idx)
            target_vol = np.sum(target == class_idx)

            if pred_vol == 0 and target_vol == 0:
                return 1.0

            return 1.0 - abs(pred_vol - target_vol) / (pred_vol + target_vol + 1e-8)

        vs_scores = []
        for i in range(1, self.num_classes):  # Skip background
            vs_scores.append(self.volumetric_similarity(pred, target, i))

        return np.array(vs_scores)

    def relative_absolute_volume_difference(
        self, pred: np.ndarray, target: np.ndarray, class_idx: int = None
    ) -> Union[float, np.ndarray]:
        """
        Calculate Relative Absolute Volume Difference (RAVD).

        RAVD = |V_pred - V_target| / V_target

        Args:
            pred: Predicted segmentation mask
            target: Ground truth segmentation mask
            class_idx: Specific class to calculate (None for all classes)

        Returns:
            RAVD value(s)
        """
        if class_idx is not None:
            pred_vol = np.sum(pred == class_idx)
            target_vol = np.sum(target == class_idx)

            if target_vol == 0:
                return float("inf") if pred_vol > 0 else 0.0

            return abs(pred_vol - target_vol) / target_vol

        ravd_scores = []
        for i in range(1, self.num_classes):  # Skip background
            ravd_scores.append(self.relative_absolute_volume_difference(pred, target, i))

        return np.array(ravd_scores)

    def calculate_all_metrics(self, pred: np.ndarray, target: np.ndarray) -> dict[str, dict[str, float]]:
        """
        Calculate comprehensive set of medical segmentation metrics.

        Args:
            pred: Predicted segmentation mask
            target: Ground truth segmentation mask

        Returns:
            Dictionary containing all metrics for each class
        """
        metrics = {}

        # Calculate metrics for each class
        for i, class_name in enumerate(self.class_names[1:], 1):  # Skip background
            metrics[class_name] = {
                "dice": float(self.dice_coefficient(pred, target, i)),
                "jaccard": float(self.jaccard_index(pred, target, i)),
                "sensitivity": float(self.sensitivity_recall(pred, target, i)),
                "specificity": float(self.specificity(pred, target, i)),
                "hausdorff_95": float(self.hausdorff_distance(pred, target, i, 95)),
                "avg_surface_distance": float(self.average_surface_distance(pred, target, i)),
                "volumetric_similarity": float(self.volumetric_similarity(pred, target, i)),
                "ravd": float(self.relative_absolute_volume_difference(pred, target, i)),
            }

        # Calculate overall metrics
        dice_scores = self.dice_coefficient(pred, target)
        metrics["Overall"] = {
            "mean_dice": float(np.mean(dice_scores)),
            "std_dice": float(np.std(dice_scores)),
            "median_dice": float(np.median(dice_scores)),
        }

        return metrics

    def brats_challenge_metrics(self, pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
        """
        Calculate metrics used in BraTS Challenge for standardized comparison.

        BraTS uses specific tumor region definitions:
        - Whole Tumor (WT): Union of all tumor classes
        - Tumor Core (TC): Core + Enhancing regions
        - Enhancing Tumor (ET): Enhancing region only

        Args:
            pred: Predicted segmentation mask
            target: Ground truth segmentation mask

        Returns:
            BraTS-style metrics
        """
        # Define BraTS regions (assuming classes: 0=background, 1=core, 2=edema, 3=enhancing)
        wt_pred = (pred > 0).astype(np.uint8)  # Whole tumor
        wt_target = (target > 0).astype(np.uint8)

        tc_pred = ((pred == 1) | (pred == 3)).astype(np.uint8)  # Tumor core
        tc_target = ((target == 1) | (target == 3)).astype(np.uint8)

        et_pred = (pred == 3).astype(np.uint8)  # Enhancing tumor
        et_target = (target == 3).astype(np.uint8)

        # Calculate metrics for each region
        brats_metrics = {}

        regions = {"WT": (wt_pred, wt_target), "TC": (tc_pred, tc_target), "ET": (et_pred, et_target)}

        for region_name, (pred_region, target_region) in regions.items():
            # Dice
            intersection = np.sum(pred_region * target_region)
            total = np.sum(pred_region) + np.sum(target_region)
            dice = 2.0 * intersection / (total + 1e-8)

            # Sensitivity
            tp = intersection
            fn = np.sum((1 - pred_region) * target_region)
            sensitivity = tp / (tp + fn + 1e-8)

            # Specificity
            tn = np.sum((1 - pred_region) * (1 - target_region))
            fp = np.sum(pred_region * (1 - target_region))
            specificity = tn / (tn + fp + 1e-8)

            # Hausdorff 95
            try:
                if np.sum(pred_region) > 0 and np.sum(target_region) > 0:
                    pred_surface = self._get_surface_points(pred_region)
                    target_surface = self._get_surface_points(target_region)

                    if len(pred_surface) > 0 and len(target_surface) > 0:
                        distances = np.concatenate(
                            [
                                np.min(np.linalg.norm(pred_surface[:, None] - target_surface, axis=2), axis=1),
                                np.min(np.linalg.norm(target_surface[:, None] - pred_surface, axis=2), axis=1),
                            ]
                        )
                        hd95 = np.percentile(distances, 95)
                    else:
                        hd95 = float("inf")
                else:
                    hd95 = float("inf")
            except:
                hd95 = float("inf")

            brats_metrics[f"{region_name}_Dice"] = dice
            brats_metrics[f"{region_name}_Sensitivity"] = sensitivity
            brats_metrics[f"{region_name}_Specificity"] = specificity
            brats_metrics[f"{region_name}_Hausdorff95"] = hd95

        return brats_metrics

    def statistical_significance_test(
        self, results1: list[dict], results2: list[dict], metric: str = "dice"
    ) -> dict[str, float]:
        """
        Perform statistical significance testing between two methods.

        Uses paired t-test for comparing segmentation results.

        Args:
            results1: Results from method 1
            results2: Results from method 2
            metric: Metric to compare

        Returns:
            Statistical test results
        """
        from scipy.stats import ttest_rel, wilcoxon

        values1 = [r.get(metric, 0) for r in results1]
        values2 = [r.get(metric, 0) for r in results2]

        # Paired t-test
        t_stat, t_pvalue = ttest_rel(values1, values2)

        # Wilcoxon signed-rank test (non-parametric)
        w_stat, w_pvalue = wilcoxon(values1, values2, alternative="two-sided")

        return {
            "ttest_statistic": t_stat,
            "ttest_pvalue": t_pvalue,
            "wilcoxon_statistic": w_stat,
            "wilcoxon_pvalue": w_pvalue,
            "significant_at_005": min(t_pvalue, w_pvalue) < 0.05,
            "significant_at_001": min(t_pvalue, w_pvalue) < 0.01,
        }


def main():
    """Demonstration of medical metrics calculation."""
    # Create sample data
    np.random.seed(42)
    pred = np.random.randint(0, 4, (64, 64, 64))
    target = np.random.randint(0, 4, (64, 64, 64))

    # Initialize metrics calculator
    metrics_calc = MedicalSegmentationMetrics()

    # Calculate all metrics
    all_metrics = metrics_calc.calculate_all_metrics(pred, target)

    print("Medical Segmentation Metrics Results:")
    print("=" * 50)

    for class_name, metrics in all_metrics.items():
        print(f"\n{class_name}:")
        for metric_name, value in metrics.items():
            print(f"  {metric_name}: {value:.4f}")

    # BraTS challenge metrics
    brats_metrics = metrics_calc.brats_challenge_metrics(pred, target)
    print("\nBraTS Challenge Metrics:")
    print("-" * 30)
    for metric, value in brats_metrics.items():
        print(f"{metric}: {value:.4f}")


if __name__ == "__main__":
    main()
