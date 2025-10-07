#!/usr/bin/env python3
"""
BraTS Dataset Adapter for Enhanced Multimodal Framework

This module provides data loading and preprocessing capabilities specifically
designed for the BraTS 2021 dataset, integrating it with our enhanced
multimodal brain tumor segmentation framework.

BraTS Data Structure:
- T1: Native T1-weighted MRI
- T1ce: Post-contrast T1-weighted MRI (similar to CT contrast behavior)
- T2: T2-weighted MRI
- FLAIR: Fluid-attenuated inversion recovery MRI
- seg: Segmentation mask (0=background, 1=NCR/NET, 2=ED, 4=ET)

Our Framework Adaptation:
- Primary modality: T1ce (contrast-enhanced, CT-like properties)
- Secondary modality: FLAIR (good tumor visibility)
- Segmentation: Convert BraTS labels to our 4-class system
"""

import numpy as np
import nibabel as nib
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional, Union
import logging
import json
import random
from scipy import ndimage
from sklearn.model_selection import train_test_split
import cv2

# Import our framework components
try:
    from advanced_data_preprocessing import AdvancedPreprocessingPipeline, PreprocessingConfig
    from medical_metrics import MedicalSegmentationMetrics
except ImportError:
    print("Warning: Some framework components not found")


class BraTSConfig:
    """Configuration for BraTS dataset processing"""

    def __init__(self):
        # Dataset paths
        self.brats_root = "/Volumes/Seagate/数据集/BraTS_Dataset_Complete"
        self.original_data_dir = self.brats_root + "/01_Original_NIfTI_Data/data"
        self.preprocessed_dir = self.brats_root + "/02_Preprocessed_H5_Data"

        # Modality selection for our multimodal framework
        self.primary_modality = "t1ce"    # T1 contrast-enhanced (CT-like)
        self.secondary_modality = "flair"  # FLAIR (good tumor visibility)

        # Preprocessing parameters
        self.target_size = (240, 240, 155)  # Standard BraTS size
        self.resize_to = (256, 256, 128)    # Our framework size
        self.normalize_method = "zscore_robust"
        self.apply_bias_correction = True

        # Augmentation settings
        self.augment_training = True
        self.augmentation_probability = 0.3

        # Label mapping: BraTS -> Our framework
        # BraTS: 0=background, 1=NCR/NET, 2=ED, 4=ET
        # Ours: 0=background, 1=core, 2=edema, 3=enhancing
        self.label_mapping = {
            0: 0,  # Background -> Background
            1: 1,  # NCR/NET -> Core
            2: 2,  # Edema -> Edema
            4: 3   # Enhancing -> Enhancing
        }


class BraTSCase:
    """Represents a single BraTS case with all modalities"""

    def __init__(self, case_path: Path, config: BraTSConfig):
        self.case_path = Path(case_path)
        self.case_id = self.case_path.name
        self.config = config

        # File paths for all modalities
        self.modality_paths = {
            't1': self.case_path / f"{self.case_id}_t1.nii.gz",
            't1ce': self.case_path / f"{self.case_id}_t1ce.nii.gz",
            't2': self.case_path / f"{self.case_id}_t2.nii.gz",
            'flair': self.case_path / f"{self.case_id}_flair.nii.gz",
            'seg': self.case_path / f"{self.case_id}_seg.nii.gz"
        }

        # Verify all files exist
        self.valid = all(path.exists() for path in self.modality_paths.values())

    def load_modality(self, modality: str) -> np.ndarray:
        """Load a specific modality"""
        if modality not in self.modality_paths:
            raise ValueError(f"Unknown modality: {modality}")

        if not self.modality_paths[modality].exists():
            raise FileNotFoundError(f"File not found: {self.modality_paths[modality]}")

        # Load NIfTI file
        img = nib.load(str(self.modality_paths[modality]))
        data = img.get_fdata().astype(np.float32)

        return data

    def load_segmentation(self) -> np.ndarray:
        """Load and convert segmentation mask"""
        seg_data = self.load_modality('seg')

        # Convert BraTS labels to our framework labels
        converted_seg = np.zeros_like(seg_data, dtype=np.uint8)
        for brats_label, our_label in self.config.label_mapping.items():
            converted_seg[seg_data == brats_label] = our_label

        return converted_seg

    def get_multimodal_pair(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get the modality pair for our multimodal framework"""
        primary = self.load_modality(self.config.primary_modality)    # T1ce
        secondary = self.load_modality(self.config.secondary_modality) # FLAIR
        segmentation = self.load_segmentation()

        return primary, secondary, segmentation

    def get_case_info(self) -> Dict:
        """Get case metadata"""
        return {
            'case_id': self.case_id,
            'case_path': str(self.case_path),
            'modalities_available': list(self.modality_paths.keys()),
            'valid': self.valid,
            'primary_modality': self.config.primary_modality,
            'secondary_modality': self.config.secondary_modality
        }


class BraTSDatasetLoader:
    """Main dataset loader for BraTS data"""

    def __init__(self, config: BraTSConfig = None):
        self.config = config or BraTSConfig()
        self.logger = logging.getLogger(__name__)

        # Initialize preprocessing pipeline
        preprocess_config = PreprocessingConfig(
            target_size=self.config.resize_to,
            bias_field_correction=self.config.apply_bias_correction,
            normalization_method=self.config.normalize_method
        )
        self.preprocessor = AdvancedPreprocessingPipeline(preprocess_config)

        # Load case list
        self.cases = self._discover_cases()
        self.logger.info(f"Discovered {len(self.cases)} valid BraTS cases")

    def _discover_cases(self) -> List[BraTSCase]:
        """Discover all valid BraTS cases"""
        data_dir = Path(self.config.original_data_dir)

        if not data_dir.exists():
            raise FileNotFoundError(f"BraTS data directory not found: {data_dir}")

        cases = []
        case_dirs = [d for d in data_dir.iterdir() if d.is_dir() and d.name.startswith('BraTS')]

        for case_dir in sorted(case_dirs):
            case = BraTSCase(case_dir, self.config)
            if case.valid:
                cases.append(case)
            else:
                self.logger.warning(f"Invalid case (missing files): {case.case_id}")

        return cases

    def get_case_by_id(self, case_id: str) -> Optional[BraTSCase]:
        """Get a specific case by ID"""
        for case in self.cases:
            if case.case_id == case_id:
                return case
        return None

    def split_dataset(self, train_ratio: float = 0.7, val_ratio: float = 0.15,
                     test_ratio: float = 0.15, random_seed: int = 42) -> Dict[str, List[BraTSCase]]:
        """Split dataset into train/validation/test sets"""

        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
            raise ValueError("Split ratios must sum to 1.0")

        # Get case IDs for reproducible splitting
        case_ids = [case.case_id for case in self.cases]

        # First split: train vs (val + test)
        train_ids, temp_ids = train_test_split(
            case_ids,
            test_size=(val_ratio + test_ratio),
            random_state=random_seed,
            shuffle=True
        )

        # Second split: val vs test
        val_ids, test_ids = train_test_split(
            temp_ids,
            test_size=test_ratio / (val_ratio + test_ratio),
            random_state=random_seed,
            shuffle=True
        )

        # Create case lists
        splits = {
            'train': [case for case in self.cases if case.case_id in train_ids],
            'val': [case for case in self.cases if case.case_id in val_ids],
            'test': [case for case in self.cases if case.case_id in test_ids]
        }

        self.logger.info(f"Dataset split: train={len(splits['train'])}, "
                        f"val={len(splits['val'])}, test={len(splits['test'])}")

        return splits

    def preprocess_case(self, case: BraTSCase, apply_augmentation: bool = False) -> Dict[str, np.ndarray]:
        """Preprocess a single case for our framework"""

        # Load multimodal data
        primary, secondary, segmentation = case.get_multimodal_pair()

        # Extract central slices for 2D processing (for now)
        # TODO: Extend to full 3D processing
        central_slice = primary.shape[2] // 2

        primary_2d = primary[:, :, central_slice]
        secondary_2d = secondary[:, :, central_slice]
        seg_2d = segmentation[:, :, central_slice]

        # Normalize intensities
        primary_norm = self._normalize_intensity(primary_2d)
        secondary_norm = self._normalize_intensity(secondary_2d)

        # Resize to target size
        target_h, target_w = self.config.resize_to[:2]

        primary_resized = cv2.resize(primary_norm, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        secondary_resized = cv2.resize(secondary_norm, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        seg_resized = cv2.resize(seg_2d.astype(np.uint8), (target_w, target_h), interpolation=cv2.INTER_NEAREST)

        # Apply augmentation if requested
        if apply_augmentation and self.config.augment_training:
            primary_resized, secondary_resized, seg_resized = self._apply_augmentation(
                primary_resized, secondary_resized, seg_resized
            )

        return {
            'primary': primary_resized,      # T1ce (CT-like)
            'secondary': secondary_resized,  # FLAIR
            'segmentation': seg_resized,
            'case_id': case.case_id,
            'original_shape': primary.shape
        }

    def _normalize_intensity(self, image: np.ndarray, method: str = 'zscore_robust') -> np.ndarray:
        """Normalize image intensities"""

        # Remove background (assume background is close to 0)
        brain_mask = image > (image.mean() * 0.1)

        if method == 'zscore_robust':
            if np.any(brain_mask):
                brain_voxels = image[brain_mask]
                median_val = np.median(brain_voxels)
                mad = np.median(np.abs(brain_voxels - median_val))
                mad_std = mad * 1.4826  # Convert MAD to std equivalent

                normalized = (image - median_val) / (mad_std + 1e-8)
            else:
                normalized = image

        elif method == 'minmax':
            if np.any(brain_mask):
                brain_voxels = image[brain_mask]
                min_val, max_val = np.percentile(brain_voxels, [1, 99])
                normalized = (image - min_val) / (max_val - min_val + 1e-8)
                normalized = np.clip(normalized, 0, 1)
            else:
                normalized = image
        else:
            # Simple z-score
            if np.any(brain_mask):
                brain_voxels = image[brain_mask]
                mean_val = np.mean(brain_voxels)
                std_val = np.std(brain_voxels)
                normalized = (image - mean_val) / (std_val + 1e-8)
            else:
                normalized = image

        return normalized.astype(np.float32)

    def _apply_augmentation(self, primary: np.ndarray, secondary: np.ndarray,
                          seg: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Apply data augmentation"""

        if random.random() < self.config.augmentation_probability:
            # Random rotation
            angle = random.uniform(-15, 15)
            h, w = primary.shape
            center = (w // 2, h // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

            primary = cv2.warpAffine(primary, rotation_matrix, (w, h),
                                   flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            secondary = cv2.warpAffine(secondary, rotation_matrix, (w, h),
                                     flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            seg = cv2.warpAffine(seg, rotation_matrix, (w, h),
                               flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)

        if random.random() < self.config.augmentation_probability:
            # Random horizontal flip
            primary = cv2.flip(primary, 1)
            secondary = cv2.flip(secondary, 1)
            seg = cv2.flip(seg, 1)

        if random.random() < self.config.augmentation_probability:
            # Intensity augmentation
            gamma = random.uniform(0.8, 1.2)
            primary = np.power(np.clip(primary, 0, 1), gamma)
            secondary = np.power(np.clip(secondary, 0, 1), gamma)

        return primary, secondary, seg

    def create_pytorch_dataset(self, cases: List[BraTSCase], apply_augmentation: bool = False) -> 'BraTSPyTorchDataset':
        """Create PyTorch dataset from case list"""
        return BraTSPyTorchDataset(cases, self, apply_augmentation)

    def analyze_dataset_statistics(self) -> Dict:
        """Analyze dataset statistics"""
        stats = {
            'total_cases': len(self.cases),
            'modalities': list(self.cases[0].modality_paths.keys()) if self.cases else [],
            'label_distribution': {0: 0, 1: 0, 2: 0, 3: 0},  # Our 4-class system
            'image_statistics': {
                'mean_size': None,
                'intensity_ranges': {}
            }
        }

        # Sample a few cases for statistics
        sample_cases = random.sample(self.cases, min(10, len(self.cases)))

        sizes = []
        for case in sample_cases:
            try:
                primary, secondary, seg = case.get_multimodal_pair()
                sizes.append(primary.shape)

                # Count labels
                unique, counts = np.unique(seg, return_counts=True)
                for label, count in zip(unique, counts):
                    if label in stats['label_distribution']:
                        stats['label_distribution'][int(label)] += int(count)

            except Exception as e:
                self.logger.warning(f"Error analyzing case {case.case_id}: {e}")

        if sizes:
            stats['image_statistics']['mean_size'] = tuple(np.mean(sizes, axis=0).astype(int))

        return stats


class BraTSPyTorchDataset(Dataset):
    """PyTorch Dataset for BraTS data"""

    def __init__(self, cases: List[BraTSCase], loader: BraTSDatasetLoader, apply_augmentation: bool = False):
        self.cases = cases
        self.loader = loader
        self.apply_augmentation = apply_augmentation

    def __len__(self) -> int:
        return len(self.cases)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        case = self.cases[idx]

        # Preprocess case
        processed = self.loader.preprocess_case(case, self.apply_augmentation)

        # Convert to tensors
        primary_tensor = torch.from_numpy(processed['primary']).float().unsqueeze(0)  # Add channel dim
        secondary_tensor = torch.from_numpy(processed['secondary']).float().unsqueeze(0)
        seg_tensor = torch.from_numpy(processed['segmentation']).long()

        return {
            'ct': primary_tensor,      # T1ce acts as "CT" modality
            'mri': secondary_tensor,   # FLAIR acts as "MRI" modality
            'mask': seg_tensor,
            'case_id': processed['case_id']
        }


def create_brats_dataloaders(config: BraTSConfig = None, batch_size: int = 4,
                           num_workers: int = 2) -> Dict[str, DataLoader]:
    """Create DataLoaders for train/val/test splits"""

    # Initialize loader
    loader = BraTSDatasetLoader(config)

    # Split dataset
    splits = loader.split_dataset()

    # Create PyTorch datasets
    datasets = {
        'train': loader.create_pytorch_dataset(splits['train'], apply_augmentation=True),
        'val': loader.create_pytorch_dataset(splits['val'], apply_augmentation=False),
        'test': loader.create_pytorch_dataset(splits['test'], apply_augmentation=False)
    }

    # Create DataLoaders
    dataloaders = {
        split: DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == 'train'),
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available()
        )
        for split, dataset in datasets.items()
    }

    return dataloaders


def main():
    """Test the BraTS dataset adapter"""
    print("🧠 Testing BraTS Dataset Adapter...")

    # Initialize configuration
    config = BraTSConfig()
    print(f"📁 BraTS data directory: {config.original_data_dir}")

    # Initialize dataset loader
    try:
        loader = BraTSDatasetLoader(config)
        print(f"✅ Successfully loaded {len(loader.cases)} BraTS cases")

        # Analyze dataset
        stats = loader.analyze_dataset_statistics()
        print(f"\n📊 Dataset Statistics:")
        print(f"   Total cases: {stats['total_cases']}")
        print(f"   Modalities: {stats['modalities']}")
        print(f"   Mean image size: {stats['image_statistics']['mean_size']}")
        print(f"   Label distribution: {stats['label_distribution']}")

        # Test case loading
        if loader.cases:
            test_case = loader.cases[0]
            print(f"\n🔍 Testing case: {test_case.case_id}")

            # Load multimodal data
            primary, secondary, seg = test_case.get_multimodal_pair()
            print(f"   Primary (T1ce) shape: {primary.shape}")
            print(f"   Secondary (FLAIR) shape: {secondary.shape}")
            print(f"   Segmentation shape: {seg.shape}")
            print(f"   Unique labels: {np.unique(seg)}")

            # Test preprocessing
            processed = loader.preprocess_case(test_case)
            print(f"   Processed primary shape: {processed['primary'].shape}")
            print(f"   Processed secondary shape: {processed['secondary'].shape}")
            print(f"   Processed segmentation shape: {processed['segmentation'].shape}")

        # Test DataLoader creation
        print(f"\n🔄 Testing DataLoader creation...")
        dataloaders = create_brats_dataloaders(config, batch_size=2)

        # Test a batch
        train_loader = dataloaders['train']
        batch = next(iter(train_loader))

        print(f"   Batch CT shape: {batch['ct'].shape}")
        print(f"   Batch MRI shape: {batch['mri'].shape}")
        print(f"   Batch mask shape: {batch['mask'].shape}")
        print(f"   Case IDs: {batch['case_id']}")

        print(f"\n✅ BraTS Dataset Adapter test completed successfully!")
        return True

    except Exception as e:
        print(f"❌ Error testing BraTS adapter: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    if success:
        print("\n🚀 Ready to integrate with enhanced multimodal framework!")
    else:
        print("\n⚠️  Please check the BraTS dataset path and structure.")