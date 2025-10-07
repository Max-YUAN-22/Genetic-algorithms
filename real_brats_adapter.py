#!/usr/bin/env python3
"""
Real BraTS 2021 Dataset Adapter for SCI Q2+ Publication

This module provides REAL NIfTI data loading for the BraTS 2021 dataset
without any mock or simulation components. This is essential for
SCI publication quality research.

Author: Research Team
Purpose: SCI Q2+ Publication in Medical Image Analysis
"""

import numpy as np
import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

try:
    import nibabel as nib
    import SimpleITK as sitk
    from skimage import transform
    MEDICAL_LIBS_AVAILABLE = True
except ImportError:
    MEDICAL_LIBS_AVAILABLE = False
    print("⚠️  Medical imaging libraries not installed!")
    print("Please run: pip install nibabel SimpleITK scikit-image")


class RealBraTSConfig:
    """Real BraTS 2021 dataset configuration for SCI publication"""

    def __init__(self):
        # Dataset paths - REAL DATA ONLY
        self.brats_root = "/Volumes/Seagate/数据集/BraTS_Dataset_Complete"
        self.original_data_dir = os.path.join(self.brats_root, "01_Original_NIfTI_Data", "data")

        # Modality selection based on medical literature
        # T1ce: Best for enhancing tumor visualization
        # FLAIR: Best for edema and infiltrative regions
        self.primary_modality = "t1ce"    # Contrast-enhanced T1
        self.secondary_modality = "flair"  # FLAIR sequence

        # Standard BraTS preprocessing parameters (match YOLO model expectations)
        self.target_size = (256, 256)  # Match multimodal YOLO expectations
        self.target_spacing = (1.0, 1.0, 1.0)  # mm

        # BraTS 2021 official label mapping
        self.brats_labels = {
            0: "Background",
            1: "Necrotic and Non-Enhancing Tumor Core (NCR/NET)",
            2: "Peritumoral Edema",
            4: "GD-Enhancing Tumor"  # Note: label 3 doesn't exist in BraTS
        }

        # Our framework label mapping (sequential)
        self.label_mapping = {
            0: 0,  # Background
            1: 1,  # NCR/NET -> Core
            2: 2,  # Edema -> Edema
            4: 3   # Enhancing -> Enhancing
        }

        # BraTS challenge evaluation regions
        self.evaluation_regions = {
            'WT': [1, 2, 4],    # Whole Tumor
            'TC': [1, 4],       # Tumor Core
            'ET': [4]           # Enhancing Tumor
        }


class RealBraTSLoader:
    """Real BraTS 2021 data loader - NO MOCK DATA"""

    def __init__(self, config: RealBraTSConfig = None):
        if not MEDICAL_LIBS_AVAILABLE:
            raise ImportError("Medical imaging libraries required for real data loading!")

        self.config = config or RealBraTSConfig()
        self.logger = logging.getLogger(__name__)

        # Verify real data availability
        if not Path(self.config.original_data_dir).exists():
            raise FileNotFoundError(f"BraTS data not found at: {self.config.original_data_dir}")

        # Discover real cases
        self.case_list = self._discover_real_cases()
        self.logger.info(f"✅ Found {len(self.case_list)} real BraTS cases")

        if len(self.case_list) == 0:
            raise ValueError("No real BraTS cases found!")

    def _discover_real_cases(self) -> List[str]:
        """Discover all real BraTS case directories"""
        data_dir = Path(self.config.original_data_dir)

        if not data_dir.exists():
            return []

        case_dirs = []
        for item in data_dir.iterdir():
            if item.is_dir() and item.name.startswith('BraTS'):
                # Verify this case has all required files
                if self._verify_case_files(item):
                    case_dirs.append(item.name)

        return sorted(case_dirs)

    def _verify_case_files(self, case_dir: Path) -> bool:
        """Verify a case has all required modalities and segmentation"""
        required_files = [
            f"{case_dir.name}_t1ce.nii.gz",    # T1 contrast enhanced
            f"{case_dir.name}_flair.nii.gz",   # FLAIR
            f"{case_dir.name}_seg.nii.gz"      # Segmentation
        ]

        for required_file in required_files:
            if not (case_dir / required_file).exists():
                return False

        return True

    def load_real_case(self, case_id: str) -> Dict[str, np.ndarray]:
        """Load a real BraTS case from NIfTI files"""
        case_dir = Path(self.config.original_data_dir) / case_id

        if not case_dir.exists():
            raise FileNotFoundError(f"Case directory not found: {case_dir}")

        try:
            # Load T1ce (primary modality)
            t1ce_path = case_dir / f"{case_id}_t1ce.nii.gz"
            t1ce_nii = nib.load(str(t1ce_path))
            t1ce_data = t1ce_nii.get_fdata()

            # Load FLAIR (secondary modality)
            flair_path = case_dir / f"{case_id}_flair.nii.gz"
            flair_nii = nib.load(str(flair_path))
            flair_data = flair_nii.get_fdata()

            # Load segmentation
            seg_path = case_dir / f"{case_id}_seg.nii.gz"
            seg_nii = nib.load(str(seg_path))
            seg_data = seg_nii.get_fdata()

            # Get middle slice (most informative for 2D analysis)
            middle_slice = seg_data.shape[2] // 2

            # Extract 2D slices
            t1ce_slice = t1ce_data[:, :, middle_slice]
            flair_slice = flair_data[:, :, middle_slice]
            seg_slice = seg_data[:, :, middle_slice]

            # Resize to target size
            t1ce_slice = self._resize_slice(t1ce_slice, self.config.target_size)
            flair_slice = self._resize_slice(flair_slice, self.config.target_size)
            seg_slice = self._resize_slice(seg_slice, self.config.target_size, is_label=True)

            # Apply BraTS label mapping
            seg_slice = self._apply_label_mapping(seg_slice)

            # Normalize intensities
            t1ce_slice = self._normalize_intensity(t1ce_slice)
            flair_slice = self._normalize_intensity(flair_slice)

            return {
                't1ce': t1ce_slice.astype(np.float32),
                'flair': flair_slice.astype(np.float32),
                'seg': seg_slice.astype(np.uint8),
                'case_id': case_id,
                'slice_index': middle_slice,
                'original_shape': t1ce_data.shape
            }

        except Exception as e:
            self.logger.error(f"Failed to load real case {case_id}: {e}")
            raise

    def _resize_slice(self, slice_data: np.ndarray, target_size: Tuple[int, int],
                     is_label: bool = False) -> np.ndarray:
        """Resize slice to target size"""
        if slice_data.shape[:2] == target_size:
            return slice_data

        # Use appropriate interpolation
        order = 0 if is_label else 1  # Nearest neighbor for labels, linear for images

        resized = transform.resize(
            slice_data,
            target_size,
            order=order,
            preserve_range=True,
            anti_aliasing=False if is_label else True
        )

        return resized

    def _apply_label_mapping(self, seg_data: np.ndarray) -> np.ndarray:
        """Apply BraTS to framework label mapping"""
        mapped_seg = np.zeros_like(seg_data, dtype=np.uint8)

        for brats_label, framework_label in self.config.label_mapping.items():
            mapped_seg[seg_data == brats_label] = framework_label

        return mapped_seg

    def _normalize_intensity(self, image_data: np.ndarray) -> np.ndarray:
        """Normalize image intensity using percentile normalization"""
        # Remove outliers using percentile clipping
        p1, p99 = np.percentile(image_data[image_data > 0], [1, 99])

        # Clip and normalize
        normalized = np.clip(image_data, p1, p99)

        if p99 > p1:
            normalized = (normalized - p1) / (p99 - p1)
        else:
            normalized = normalized * 0  # Handle edge case

        return normalized

    def prepare_for_framework(self, case_data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Prepare real BraTS data for multimodal framework"""
        return {
            'ct': case_data['t1ce'],      # T1ce acts as CT modality
            'mri': case_data['flair'],    # FLAIR acts as MRI modality
            'mask': case_data['seg'],     # Ground truth segmentation
            'case_id': case_data['case_id']
        }

    def get_dataset_splits(self, train_ratio: float = 0.7) -> Dict[str, List[str]]:
        """Get train/validation/test splits for real data"""
        total_cases = len(self.case_list)
        train_size = int(total_cases * train_ratio)
        val_size = int(total_cases * 0.15)

        # Deterministic splits for reproducibility
        np.random.seed(42)
        shuffled_cases = np.random.permutation(self.case_list).tolist()

        splits = {
            'train': shuffled_cases[:train_size],
            'val': shuffled_cases[train_size:train_size + val_size],
            'test': shuffled_cases[train_size + val_size:]
        }

        self.logger.info(f"Real dataset splits: train={len(splits['train'])}, "
                        f"val={len(splits['val'])}, test={len(splits['test'])}")

        return splits

    def analyze_real_dataset(self) -> Dict:
        """Analyze the real BraTS dataset"""
        self.logger.info("🔍 Analyzing real BraTS dataset...")

        # Sample analysis on first 10 cases
        sample_cases = self.case_list[:min(10, len(self.case_list))]

        stats = {
            'total_cases': len(self.case_list),
            'data_type': 'REAL NIfTI DATA',
            'modalities': ['T1ce', 'FLAIR'],
            'label_distribution': {},
            'intensity_stats': {},
            'sample_analysis': {}
        }

        label_counts = {0: 0, 1: 0, 2: 0, 3: 0}
        intensity_stats = {'t1ce': [], 'flair': []}

        for case_id in sample_cases:
            try:
                case_data = self.load_real_case(case_id)

                # Label distribution
                unique, counts = np.unique(case_data['seg'], return_counts=True)
                for label, count in zip(unique, counts):
                    if label in label_counts:
                        label_counts[label] += count

                # Intensity statistics
                intensity_stats['t1ce'].append({
                    'mean': float(np.mean(case_data['t1ce'])),
                    'std': float(np.std(case_data['t1ce'])),
                    'min': float(np.min(case_data['t1ce'])),
                    'max': float(np.max(case_data['t1ce']))
                })

                intensity_stats['flair'].append({
                    'mean': float(np.mean(case_data['flair'])),
                    'std': float(np.std(case_data['flair'])),
                    'min': float(np.min(case_data['flair'])),
                    'max': float(np.max(case_data['flair']))
                })

                # Sample case info
                stats['sample_analysis'][case_id] = {
                    'shape': case_data['t1ce'].shape,
                    'unique_labels': np.unique(case_data['seg']).tolist(),
                    'tumor_volume_ratio': float(np.sum(case_data['seg'] > 0) / case_data['seg'].size)
                }

            except Exception as e:
                self.logger.error(f"Error analyzing case {case_id}: {e}")

        stats['label_distribution'] = label_counts
        stats['intensity_stats'] = intensity_stats

        return stats

    def create_real_dataset(self, case_ids: List[str]) -> List[Dict[str, np.ndarray]]:
        """Create dataset from real BraTS cases"""
        dataset = []

        self.logger.info(f"📊 Loading {len(case_ids)} real BraTS cases...")

        for i, case_id in enumerate(case_ids):
            if i % 50 == 0:
                self.logger.info(f"Progress: {i}/{len(case_ids)} cases loaded")

            try:
                case_data = self.load_real_case(case_id)
                framework_data = self.prepare_for_framework(case_data)
                dataset.append(framework_data)

            except Exception as e:
                self.logger.error(f"Failed to load case {case_id}: {e}")

        self.logger.info(f"✅ Successfully loaded {len(dataset)} real cases")
        return dataset


def test_real_brats_loading():
    """Test real BraTS data loading"""
    print("🧠 Testing Real BraTS 2021 Data Loading...")

    if not MEDICAL_LIBS_AVAILABLE:
        print("❌ Please install: pip install nibabel SimpleITK scikit-image")
        return False

    try:
        # Initialize real loader
        config = RealBraTSConfig()
        loader = RealBraTSLoader(config)

        print(f"✅ Found {len(loader.case_list)} real BraTS cases")

        # Test loading first case
        if loader.case_list:
            first_case = loader.case_list[0]
            case_data = loader.load_real_case(first_case)

            print(f"📊 Loaded real case: {first_case}")
            print(f"   T1ce shape: {case_data['t1ce'].shape}")
            print(f"   FLAIR shape: {case_data['flair'].shape}")
            print(f"   Segmentation shape: {case_data['seg'].shape}")
            print(f"   Labels present: {np.unique(case_data['seg'])}")
            print(f"   Tumor volume: {np.sum(case_data['seg'] > 0) / case_data['seg'].size * 100:.1f}%")

            # Test framework preparation
            framework_data = loader.prepare_for_framework(case_data)
            print(f"✅ Framework data prepared successfully")

            return True
        else:
            print("❌ No BraTS cases found")
            return False

    except Exception as e:
        print(f"❌ Error: {e}")
        return False


if __name__ == "__main__":
    success = test_real_brats_loading()

    if success:
        print("\n🎉 Real BraTS data loading successful!")
        print("Ready for SCI Q2+ quality research!")
    else:
        print("\n❌ Real BraTS data loading failed!")
        print("Please check data path and install required libraries.")