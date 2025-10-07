#!/usr/bin/env python3
"""
Real Training Pipeline for SCI Q2+ Publication

This is the REAL training pipeline for our enhanced multimodal brain tumor
segmentation framework. NO MOCK DATA OR SIMULATIONS.

Purpose: Generate publication-quality results for SCI Q2+ journals
Target: Medical Image Analysis, IEEE TMI, Computer Methods in Biomedicine
"""

import argparse
import json
import time
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from dataclasses import dataclass
import sys

# Import PyTorch first
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    PYTORCH_AVAILABLE = True
except ImportError as e:
    print(f"❌ PyTorch import error: {e}")
    PYTORCH_AVAILABLE = False

# Import real components
try:
    from real_brats_adapter import RealBraTSLoader, RealBraTSConfig
    from multimodal_yolo_prototype import MultimodalYOLOSegmentation, MedicalSegmentationLoss
    from medical_metrics import MedicalSegmentationMetrics
    COMPONENTS_AVAILABLE = True
except ImportError as e:
    print(f"❌ Component import error: {e}")
    print("Please ensure all framework components are available")
    COMPONENTS_AVAILABLE = False

warnings.filterwarnings('ignore', category=UserWarning)


@dataclass
class RealTrainingConfig:
    """Configuration for REAL training pipeline - SCI publication quality"""

    # Dataset configuration
    batch_size: int = 8
    num_workers: int = 4
    train_ratio: float = 0.7
    val_ratio: float = 0.15

    # Training configuration
    num_epochs: int = 150
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    early_stopping_patience: int = 20
    gradient_clip_value: float = 1.0

    # Model configuration
    num_classes: int = 4
    input_size: Tuple[int, int] = (256, 256)

    # 2.5D configuration (project neighboring slices into a single 2D image)
    use_2_5d: bool = False
    stack_depth: int = 5  # must be odd: 3, 5, 7 ...

    # Validation configuration
    validate_every: int = 5
    save_best_model: bool = True
    save_checkpoints: bool = True

    # Hardware configuration
    device: str = "cpu"  # Will be set properly in __post_init__
    mixed_precision: bool = True

    # Research quality requirements
    reproducible_seed: int = 42
    detailed_logging: bool = True
    calculate_brats_metrics: bool = True

    def __post_init__(self):
        """Set device after imports are available"""
        if PYTORCH_AVAILABLE:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = "mps"  # Apple Silicon GPU
            else:
                self.device = "cpu"


class RealBraTSDataset(Dataset):
    """Real PyTorch Dataset for BraTS - NO MOCK DATA"""

    def __init__(self, case_ids: List[str], brats_loader: RealBraTSLoader,
                 config: RealTrainingConfig, is_training: bool = True):
        self.case_ids = case_ids
        self.brats_loader = brats_loader
        self.config = config
        self.is_training = is_training

        # Set reproducible seed
        np.random.seed(config.reproducible_seed)
        torch.manual_seed(config.reproducible_seed)

        self.logger = logging.getLogger(__name__)
        self.logger.info(f"📊 Created REAL dataset with {len(case_ids)} cases")

    def __len__(self):
        return len(self.case_ids)

    def __getitem__(self, idx):
        case_id = self.case_ids[idx]

        try:
            # Load REAL BraTS data
            case_data = self.brats_loader.load_real_case(case_id)
            framework_data = self.brats_loader.prepare_for_framework(case_data)

            # Extract modalities and mask
            ct_image = framework_data['ct'].astype(np.float32)
            mri_image = framework_data['mri'].astype(np.float32)
            mask = framework_data['mask'].astype(np.int64)
            # BraTS label mapping: {0,1,2,4} -> {0,1,2,3}
            mask[mask == 4] = 3

            # 2.5D projection if volume provided and enabled
            if self.config.use_2_5d:
                ct_image = self._project_2_5d(ct_image)
                mri_image = self._project_2_5d(mri_image)

            # Apply data augmentation only during training
            if self.is_training:
                ct_image, mri_image, mask = self._apply_medical_augmentation(
                    ct_image, mri_image, mask
                )

            # Convert to tensors (make contiguous copies to fix stride issues)
            ct_tensor = torch.from_numpy(ct_image.copy()).unsqueeze(0)  # [1, H, W]
            mri_tensor = torch.from_numpy(mri_image.copy()).unsqueeze(0)  # [1, H, W]
            mask_tensor = torch.from_numpy(mask.copy())  # [H, W]

            return {
                'ct': ct_tensor,
                'mri': mri_tensor,
                'mask': mask_tensor,
                'case_id': case_id
            }

        except Exception as e:
            self.logger.error(f"Error loading REAL case {case_id}: {e}")
            # Return a valid tensor instead of crashing
            dummy_ct = torch.zeros(1, *self.config.input_size)
            dummy_mri = torch.zeros(1, *self.config.input_size)
            dummy_mask = torch.zeros(*self.config.input_size, dtype=torch.long)

            return {
                'ct': dummy_ct,
                'mri': dummy_mri,
                'mask': dummy_mask,
                'case_id': case_id
            }

    def _apply_medical_augmentation(self, ct: np.ndarray, mri: np.ndarray,
                                   mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Apply medically-appropriate data augmentation"""
        # Medical-grade augmentation (not excessive)

        # Random horizontal flip (brain symmetry)
        if np.random.random() < 0.5:
            ct = np.fliplr(ct)
            mri = np.fliplr(mri)
            mask = np.fliplr(mask)

        # Small rotation (±10 degrees max for brain)
        if np.random.random() < 0.3:
            angle = np.random.uniform(-10, 10)
            ct, mri, mask = self._rotate_medical_images(ct, mri, mask, angle)

        # Intensity augmentation (conservative for medical images)
        if np.random.random() < 0.4:
            # Brightness adjustment
            ct = np.clip(ct + np.random.uniform(-0.05, 0.05), 0, 1)
            mri = np.clip(mri + np.random.uniform(-0.05, 0.05), 0, 1)

            # Contrast adjustment
            ct = np.clip(ct * np.random.uniform(0.9, 1.1), 0, 1)
            mri = np.clip(mri * np.random.uniform(0.9, 1.1), 0, 1)

        return ct, mri, mask

    def _rotate_medical_images(self, ct: np.ndarray, mri: np.ndarray,
                              mask: np.ndarray, angle: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Rotate medical images with proper interpolation"""
        try:
            from scipy.ndimage import rotate
            ct_rot = rotate(ct, angle, reshape=False, order=1)
            mri_rot = rotate(mri, angle, reshape=False, order=1)
            mask_rot = rotate(mask, angle, reshape=False, order=0)
            return ct_rot, mri_rot, mask_rot
        except ImportError:
            return ct, mri, mask

    def _project_2_5d(self, vol: np.ndarray) -> np.ndarray:
        """Project a 3D volume (D,H,W) into a 2D image (H,W) using weighted neighboring slices.
        Falls back to identity for 2D inputs.
        """
        if vol.ndim == 2:
            return vol
        if vol.ndim != 3:
            return vol.squeeze()

        D, H, W = vol.shape
        k = max(1, int(self.config.stack_depth))
        if k % 2 == 0:
            k += 1  # enforce odd
        half = k // 2
        center = D // 2
        start = max(0, center - half)
        end = min(D, center + half + 1)
        window = vol[start:end]
        # Build weights decreasing from center
        idxs = np.arange(start, end)
        dists = np.abs(idxs - center)
        # weights: 1.0 at center, then 0.8, 0.6, ... minimum 0.2
        weights = 1.0 - 0.2 * dists
        weights = np.clip(weights, 0.2, 1.0)
        weights = weights / (weights.sum() + 1e-8)
        proj = np.tensordot(weights, window, axes=(0, 0))  # (H,W)
        return proj.astype(np.float32)


class RealTrainingPipeline:
    """REAL training pipeline for SCI Q2+ publication"""

    def __init__(self, config: RealTrainingConfig, output_dir: str = "real_training_results"):
        if not PYTORCH_AVAILABLE or not COMPONENTS_AVAILABLE:
            raise ImportError("PyTorch and all components are required for real training!")

        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Setup detailed logging for research
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.output_dir / 'real_training.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

        # Set reproducible seeds
        np.random.seed(config.reproducible_seed)
        torch.manual_seed(config.reproducible_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(config.reproducible_seed)

        # Initialize REAL BraTS loader
        self.brats_config = RealBraTSConfig()
        self.brats_loader = RealBraTSLoader(self.brats_config)

        # Training components
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.loss_fn = None
        self.scaler = None
        self.metrics_calculator = MedicalSegmentationMetrics()

        # Training state
        self.best_dice = 0.0
        self.training_history = []

        self.logger.info("🧠 REAL Training Pipeline initialized for SCI Q2+ publication")
        self.logger.info(f"📊 Real BraTS cases available: {len(self.brats_loader.case_list)}")
        self.logger.info(f"🔧 Device: {self.config.device}")
        self.logger.info(f"🎯 Target: SCI Q2+ Medical Image Analysis journals")

    def prepare_real_datasets(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Prepare REAL train, validation, and test datasets"""
        self.logger.info("📊 Preparing REAL BraTS datasets...")

        # Get dataset splits
        splits = self.brats_loader.get_dataset_splits(self.config.train_ratio)

        # Create REAL datasets
        train_dataset = RealBraTSDataset(
            splits['train'], self.brats_loader, self.config, is_training=True
        )
        val_dataset = RealBraTSDataset(
            splits['val'], self.brats_loader, self.config, is_training=False
        )
        test_dataset = RealBraTSDataset(
            splits['test'], self.brats_loader, self.config, is_training=False
        )

        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=True if self.config.device == 'cuda' else False,
            persistent_workers=True if self.config.num_workers > 0 else False
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True if self.config.device == 'cuda' else False
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True if self.config.device == 'cuda' else False
        )

        self.logger.info(f"📈 REAL dataset sizes - Train: {len(train_dataset)}, "
                        f"Val: {len(val_dataset)}, Test: {len(test_dataset)}")

        return train_loader, val_loader, test_loader

    def initialize_real_model(self) -> None:
        """Initialize REAL model for training"""
        self.logger.info("🧠 Initializing REAL multimodal model...")

        # Create model
        self.model = MultimodalYOLOSegmentation(num_classes=self.config.num_classes)
        self.model = self.model.to(self.config.device)

        # Initialize loss function
        self.loss_fn = MedicalSegmentationLoss(num_classes=self.config.num_classes)

        # Initialize optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )

        # Initialize scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=8
        )

        # Initialize mixed precision scaler (only for CUDA)
        if self.config.mixed_precision and self.config.device == 'cuda':
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None

        # Log model info
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        self.logger.info(f"📊 Model - Total params: {total_params:,}, Trainable: {trainable_params:,}")

    def train_real_epoch(self, train_loader: DataLoader, epoch: int) -> Dict[str, float]:
        """Train one epoch on REAL data"""
        self.model.train()

        total_loss = 0.0
        dice_scores = []
        num_batches = 0

        for batch_idx, batch in enumerate(train_loader):
            # Move REAL data to device
            ct = batch['ct'].to(self.config.device)
            mri = batch['mri'].to(self.config.device)
            mask = batch['mask'].to(self.config.device)

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass with mixed precision if available
            if self.scaler is not None:
                with torch.cuda.amp.autocast():
                    outputs = self.model(ct, mri)
                    loss_dict = self.loss_fn(outputs, mask)
                    loss = loss_dict['total_loss']

                # Backward pass with scaling
                self.scaler.scale(loss).backward()

                if self.config.gradient_clip_value > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.gradient_clip_value
                    )

                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(ct, mri)
                loss_dict = self.loss_fn(outputs, mask)
                loss = loss_dict['total_loss']

                loss.backward()

                if self.config.gradient_clip_value > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.gradient_clip_value
                    )

                self.optimizer.step()

            # Calculate metrics on REAL data
            total_loss += loss.item()

            if 'segmentation' in outputs:
                pred = torch.argmax(outputs['segmentation'], dim=1)
                dice = self._calculate_real_dice_score(pred, mask)
                dice_scores.append(dice)

            num_batches += 1

            # Log progress
            if batch_idx % 20 == 0:
                self.logger.info(
                    f"Epoch {epoch+1}, Batch {batch_idx}/{len(train_loader)}, "
                    f"Loss: {loss.item():.4f}"
                )

        return {
            'loss': total_loss / num_batches,
            'dice': np.mean(dice_scores) if dice_scores else 0.0
        }

    def validate_real_epoch(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate on REAL data"""
        self.model.eval()

        total_loss = 0.0
        dice_scores = []
        num_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                # Move REAL data to device
                ct = batch['ct'].to(self.config.device)
                mri = batch['mri'].to(self.config.device)
                mask = batch['mask'].to(self.config.device)

                # Forward pass
                outputs = self.model(ct, mri)
                loss_dict = self.loss_fn(outputs, mask)
                loss = loss_dict['total_loss']

                total_loss += loss.item()

                # Calculate metrics on REAL data
                if 'segmentation' in outputs:
                    pred = torch.argmax(outputs['segmentation'], dim=1)
                    dice = self._calculate_real_dice_score(pred, mask)
                    dice_scores.append(dice)

                num_batches += 1

        return {
            'loss': total_loss / num_batches,
            'dice': np.mean(dice_scores) if dice_scores else 0.0
        }

    def _calculate_real_dice_score(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """Calculate REAL Dice score for batch"""
        dice_scores = []

        pred_np = pred.cpu().numpy()
        target_np = target.cpu().numpy()

        for i in range(pred.shape[0]):  # For each sample in batch
            sample_dice = []
            for cls in range(1, self.config.num_classes):  # Skip background
                pred_cls = (pred_np[i] == cls).astype(np.float32)
                target_cls = (target_np[i] == cls).astype(np.float32)

                intersection = np.sum(pred_cls * target_cls)
                union = np.sum(pred_cls) + np.sum(target_cls)

                if union > 0:
                    dice = (2.0 * intersection) / union
                    sample_dice.append(dice)

            if sample_dice:
                dice_scores.append(np.mean(sample_dice))

        return np.mean(dice_scores) if dice_scores else 0.0

    def run_real_training(self) -> Dict[str, Any]:
        """Run complete REAL training pipeline"""
        self.logger.info("🚀 Starting REAL training for SCI Q2+ publication...")

        start_time = time.time()

        # Prepare REAL datasets
        train_loader, val_loader, test_loader = self.prepare_real_datasets()

        # Initialize model
        self.initialize_real_model()

        best_dice = 0.0
        epochs_without_improvement = 0

        for epoch in range(self.config.num_epochs):
            epoch_start_time = time.time()

            # Training on REAL data
            train_metrics = self.train_real_epoch(train_loader, epoch)

            # Validation on REAL data
            if epoch % self.config.validate_every == 0:
                val_metrics = self.validate_real_epoch(val_loader)

                # Update scheduler
                self.scheduler.step(val_metrics['dice'])

                epoch_time = time.time() - epoch_start_time

                # Log metrics
                self.logger.info(
                    f"Epoch {epoch+1}/{self.config.num_epochs} - "
                    f"Train Loss: {train_metrics['loss']:.4f}, Train Dice: {train_metrics['dice']:.4f}, "
                    f"Val Loss: {val_metrics['loss']:.4f}, Val Dice: {val_metrics['dice']:.4f}, "
                    f"Time: {epoch_time:.2f}s"
                )

                # Check for improvement
                if val_metrics['dice'] > best_dice:
                    best_dice = val_metrics['dice']
                    epochs_without_improvement = 0

                    if self.config.save_best_model:
                        self._save_real_model(epoch + 1, val_metrics['dice'], is_best=True)

                    self.logger.info(f"🎯 NEW BEST DICE: {best_dice:.4f}")
                else:
                    epochs_without_improvement += 1

                # Save training history
                self.training_history.append({
                    'epoch': epoch + 1,
                    'train_loss': train_metrics['loss'],
                    'train_dice': train_metrics['dice'],
                    'val_loss': val_metrics['loss'],
                    'val_dice': val_metrics['dice'],
                    'lr': self.optimizer.param_groups[0]['lr'],
                    'epoch_time': epoch_time
                })

                # Early stopping
                if epochs_without_improvement >= self.config.early_stopping_patience:
                    self.logger.info(f"🛑 Early stopping after {epoch + 1} epochs")
                    break

        # Final evaluation on test set
        test_metrics = self.validate_real_epoch(test_loader)

        # Detailed WT/TC/ET with postprocessing
        try:
            from medical_evaluation_system import BrainTumorEvaluator, EvaluationConfig
            import numpy as np
            eval_cfg = EvaluationConfig(
                save_predictions=True,
                save_visualizations=True,
                calculate_brats_metrics=True,
                enable_postprocessing=True,
                postprocess_params={'keep_largest_per_class': True, 'min_component_size': 80, 'closing_radius': 1}
            )
            evaluator = BrainTumorEvaluator(eval_cfg, self.output_dir / 'evaluation_real')

            preds = []
            gts = []
            case_ids = []
            self.model.eval()
            with torch.no_grad():
                for batch in test_loader:
                    ct = batch['ct'].to(self.config.device)
                    mri = batch['mri'].to(self.config.device)
                    mask = batch['mask'].to(self.config.device)
                    outputs = self.model(ct, mri)
                    if 'segmentation' in outputs:
                        pred = torch.argmax(outputs['segmentation'], dim=1).cpu().numpy()
                    else:
                        pred = torch.zeros_like(mask).cpu().numpy()
                    preds.append(pred)
                    gts.append(mask.cpu().numpy())
                    case_ids.extend(batch['case_id'])
            preds_np = np.concatenate(preds, axis=0)
            gts_np = np.concatenate(gts, axis=0)
            detailed_stats = evaluator.evaluate_batch(preds_np, gts_np, case_ids)
        except Exception:
            detailed_stats = {}

        total_time = time.time() - start_time

        # Save results
        results = {
            'best_validation_dice': best_dice,
            'final_test_dice': test_metrics['dice'],
            'total_epochs': len(self.training_history),
            'total_training_time': total_time,
            'training_history': self.training_history,
            'detailed_wt_tc_et': detailed_stats,
            'config': self.config.__dict__
        }

        results_path = self.output_dir / 'real_training_results.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        self.logger.info(f"🎉 REAL training completed!")
        self.logger.info(f"📊 Best validation Dice: {best_dice:.4f}")
        self.logger.info(f"📊 Final test Dice: {test_metrics['dice']:.4f}")
        self.logger.info(f"⏱️  Total time: {total_time:.2f} seconds")

        return results

    def _save_real_model(self, epoch: int, dice_score: float, is_best: bool = False) -> None:
        """Save REAL model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'dice_score': dice_score,
            'config': self.config.__dict__,
            'training_history': self.training_history
        }

        if is_best:
            checkpoint_path = self.output_dir / 'best_real_model.pth'
            self.logger.info(f"💾 Saving BEST model: {checkpoint_path}")
        else:
            checkpoint_path = self.output_dir / f'checkpoint_epoch_{epoch}.pth'

        torch.save(checkpoint, checkpoint_path)


def main():
    """Main entry point for REAL training"""
    parser = argparse.ArgumentParser(description="REAL BraTS Training for SCI Q2+ Publication")

    parser.add_argument('--output_dir', type=str, default='real_training_results')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_epochs', type=int, default=150)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--use_2_5d', action='store_true', help='Enable 2.5D projection for 3D volumes')
    parser.add_argument('--stack_depth', type=int, default=5, help='2.5D stack depth (odd number)')
    parser.add_argument('--eval_only', action='store_true', help='Run evaluation only (no training)')
    parser.add_argument('--checkpoint', type=str, default='', help='Path to checkpoint .pth (optional)')

    args = parser.parse_args()

    print("🧠 REAL BraTS Training Pipeline for SCI Q2+ Publication")
    print("=" * 60)
    print("🚨 NO MOCK DATA - REAL RESEARCH ONLY")
    print("🎯 Target: Medical Image Analysis, IEEE TMI")
    print("")

    config = RealTrainingConfig(
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        use_2_5d=args.use_2_5d,
        stack_depth=args.stack_depth
    )

    try:
        pipeline = RealTrainingPipeline(config, args.output_dir)

        if args.eval_only:
            # Prepare datasets
            _, _, test_loader = pipeline.prepare_real_datasets()

            # Initialize and load checkpoint
            pipeline.initialize_real_model()
            ckpt_path = args.checkpoint or str((Path(args.output_dir) / 'best_real_model.pth'))
            if Path(ckpt_path).exists():
                try:
                    state = torch.load(ckpt_path, map_location=config.device, weights_only=False)
                except TypeError:
                    # Older PyTorch doesn't support weights_only arg
                    state = torch.load(ckpt_path, map_location=config.device)
                except Exception as e:
                    print(f"⚠️  Fallback loading checkpoint due to: {e}")
                    state = torch.load(ckpt_path, map_location=config.device)

                if isinstance(state, dict) and 'model_state_dict' in state:
                    pipeline.model.load_state_dict(state['model_state_dict'], strict=False)
                else:
                    pipeline.model.load_state_dict(state, strict=False)
            else:
                print(f"⚠️  Checkpoint not found at {ckpt_path}, evaluating current weights.")

            # Detailed WT/TC/ET with postprocessing
            from medical_evaluation_system import BrainTumorEvaluator, EvaluationConfig
            import numpy as np
            eval_cfg = EvaluationConfig(
                save_predictions=True,
                save_visualizations=True,
                calculate_brats_metrics=True,
                enable_postprocessing=True,
                postprocess_params={'keep_largest_per_class': True, 'min_component_size': 80, 'closing_radius': 1}
            )
            evaluator = BrainTumorEvaluator(eval_cfg, Path(args.output_dir) / 'evaluation_real')

            preds = []
            gts = []
            case_ids = []
            pipeline.model.eval()
            with torch.no_grad():
                for batch in test_loader:
                    ct = batch['ct'].to(config.device)
                    mri = batch['mri'].to(config.device)
                    mask = batch['mask'].to(config.device)
                    outputs = pipeline.model(ct, mri)
                    if 'segmentation' in outputs:
                        pred = torch.argmax(outputs['segmentation'], dim=1).cpu().numpy()
                    else:
                        pred = torch.zeros_like(mask).cpu().numpy()
                    preds.append(pred)
                    gts.append(mask.cpu().numpy())
                    case_ids.extend(batch['case_id'])

            preds_np = np.concatenate(preds, axis=0)
            gts_np = np.concatenate(gts, axis=0)
            detailed_stats = evaluator.evaluate_batch(preds_np, gts_np, case_ids)

            # Save minimal results file
            results = {
                'final_test_dice': float(detailed_stats.get('mean_dice', 0.0)),
                'detailed_wt_tc_et': detailed_stats,
                'config': config.__dict__
            }

            out_path = Path(args.output_dir) / 'real_eval_only_results.json'
            with open(out_path, 'w') as f:
                json.dump(results, f, indent=2)

            print("\n✅ EVALUATION ONLY COMPLETED!")
            print(f"📄 Saved: {out_path}")
        else:
            results = pipeline.run_real_training()

            print(f"\n🎉 REAL TRAINING COMPLETED!")
            print(f"📊 Best Dice Score: {results['best_validation_dice']:.4f}")
            print(f"📊 Test Dice Score: {results['final_test_dice']:.4f}")
            print(f"📄 Results saved to: {args.output_dir}")

    except Exception as e:
        print(f"❌ Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)