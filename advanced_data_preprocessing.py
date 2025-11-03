#!/usr/bin/env python3
"""
Advanced Data Preprocessing Pipeline for Multimodal Brain Tumor Segmentation.

This module implements sophisticated preprocessing techniques for CT and MRI brain images,
including advanced registration, normalization, and augmentation strategies specifically
designed for medical imaging applications.

Key Features:
1. Advanced image registration (mutual information-based)
2. Multi-scale normalization techniques
3. Medical image-specific augmentation
4. Quality assessment and artifact detection
5. Cross-modal intensity harmonization
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import nibabel as nib
import numpy as np
from scipy import ndimage
from scipy.optimize import minimize


@dataclass
class PreprocessingConfig:
    """Configuration for preprocessing pipeline."""

    target_spacing: tuple[float, float, float] = (1.0, 1.0, 1.0)  # mm
    target_size: tuple[int, int, int] = (128, 128, 128)
    intensity_clipping: tuple[float, float] = (0.5, 99.5)  # percentiles
    bias_field_correction: bool = True
    skull_stripping: bool = True
    registration_method: str = "mutual_information"
    normalization_method: str = "zscore_robust"
    augmentation_probability: float = 0.5


class MutualInformationRegistration:
    """Advanced mutual information-based image registration."""

    def __init__(self, bins: int = 64, learning_rate: float = 0.01, max_iterations: int = 300):
        self.bins = bins
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations

    def mutual_information(self, fixed: np.ndarray, moving: np.ndarray, transform_params: np.ndarray) -> float:
        """
        Calculate mutual information between fixed and moving images.

        MI(A,B) = H(A) + H(B) - H(A,B)
        where H is entropy
        """
        # Apply transformation
        moving_transformed = self._apply_affine_transform(moving, transform_params)

        # Compute joint histogram
        hist_2d, _, _ = np.histogram2d(fixed.ravel(), moving_transformed.ravel(), bins=self.bins)

        # Add small epsilon to avoid log(0)
        hist_2d = hist_2d + np.finfo(float).eps

        # Normalize to probability distribution
        hist_2d = hist_2d / np.sum(hist_2d)

        # Marginal distributions
        hist_fixed = np.sum(hist_2d, axis=1)
        hist_moving = np.sum(hist_2d, axis=0)

        # Calculate entropies
        entropy_fixed = -np.sum(hist_fixed * np.log2(hist_fixed + np.finfo(float).eps))
        entropy_moving = -np.sum(hist_moving * np.log2(hist_moving + np.finfo(float).eps))
        entropy_joint = -np.sum(hist_2d * np.log2(hist_2d + np.finfo(float).eps))

        # Mutual information
        mi = entropy_fixed + entropy_moving - entropy_joint
        return -mi  # Negative for minimization

    def _apply_affine_transform(self, image: np.ndarray, params: np.ndarray) -> np.ndarray:
        """Apply affine transformation to image."""
        # Extract transformation parameters
        tx, ty, tz, rx, ry, rz, sx, sy, sz = params

        # Create transformation matrices
        translation = np.array([[1, 0, 0, tx], [0, 1, 0, ty], [0, 0, 1, tz], [0, 0, 0, 1]])

        # Rotation matrices
        cos_rx, sin_rx = np.cos(rx), np.sin(rx)
        cos_ry, sin_ry = np.cos(ry), np.sin(ry)
        cos_rz, sin_rz = np.cos(rz), np.sin(rz)

        rot_x = np.array([[1, 0, 0, 0], [0, cos_rx, -sin_rx, 0], [0, sin_rx, cos_rx, 0], [0, 0, 0, 1]])

        rot_y = np.array([[cos_ry, 0, sin_ry, 0], [0, 1, 0, 0], [-sin_ry, 0, cos_ry, 0], [0, 0, 0, 1]])

        rot_z = np.array([[cos_rz, -sin_rz, 0, 0], [sin_rz, cos_rz, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

        # Scaling matrix
        scaling = np.array([[sx, 0, 0, 0], [0, sy, 0, 0], [0, 0, sz, 0], [0, 0, 0, 1]])

        # Combined transformation
        transform = translation @ rot_z @ rot_y @ rot_x @ scaling

        # Apply transformation (simplified for demonstration)
        # In practice, would use proper 3D transformation
        transformed = ndimage.affine_transform(image, transform[:3, :3], offset=transform[:3, 3])
        return transformed

    def register(self, fixed: np.ndarray, moving: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Register moving image to fixed image using mutual information.

        Returns:
            Tuple of (registered_moving_image, transformation_parameters)
        """
        # Initial parameters: [tx, ty, tz, rx, ry, rz, sx, sy, sz]
        initial_params = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1])

        # Optimization bounds
        bounds = [
            (-20, 20),
            (-20, 20),
            (-20, 20),  # translation
            (-0.2, 0.2),
            (-0.2, 0.2),
            (-0.2, 0.2),  # rotation (radians)
            (0.8, 1.2),
            (0.8, 1.2),
            (0.8, 1.2),  # scaling
        ]

        # Optimize
        result = minimize(
            lambda params: self.mutual_information(fixed, moving, params),
            initial_params,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": self.max_iterations},
        )

        # Apply final transformation
        optimal_params = result.x
        registered = self._apply_affine_transform(moving, optimal_params)

        return registered, optimal_params


class AdvancedNormalization:
    """Advanced normalization techniques for medical images."""

    @staticmethod
    def zscore_normalization(image: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
        """Z-score normalization within brain mask."""
        if mask is not None:
            brain_voxels = image[mask > 0]
            mean_val = np.mean(brain_voxels)
            std_val = np.std(brain_voxels)
        else:
            mean_val = np.mean(image)
            std_val = np.std(image)

        normalized = (image - mean_val) / (std_val + 1e-8)
        return normalized

    @staticmethod
    def robust_zscore_normalization(image: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
        """Robust Z-score using median and MAD."""
        if mask is not None:
            brain_voxels = image[mask > 0]
        else:
            brain_voxels = image.flatten()

        median_val = np.median(brain_voxels)
        mad = np.median(np.abs(brain_voxels - median_val))

        # Convert MAD to standard deviation equivalent
        mad_std = mad * 1.4826

        normalized = (image - median_val) / (mad_std + 1e-8)
        return normalized

    @staticmethod
    def histogram_matching(source: np.ndarray, reference: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
        """Match histogram of source image to reference image."""
        if mask is not None:
            source_values = source[mask > 0]
            ref_values = reference[mask > 0]
        else:
            source_values = source.flatten()
            ref_values = reference.flatten()

        # Get histograms
        source_hist, source_bins = np.histogram(source_values, bins=256, density=True)
        ref_hist, ref_bins = np.histogram(ref_values, bins=256, density=True)

        # Calculate CDFs
        source_cdf = np.cumsum(source_hist)
        ref_cdf = np.cumsum(ref_hist)

        # Normalize CDFs
        source_cdf = source_cdf / source_cdf[-1]
        ref_cdf = ref_cdf / ref_cdf[-1]

        # Interpolate to create mapping
        mapping = np.interp(source_cdf, ref_cdf, ref_bins[:-1])

        # Apply mapping
        matched = np.interp(source.flatten(), source_bins[:-1], mapping)
        return matched.reshape(source.shape)

    @staticmethod
    def nyul_normalization(images: list[np.ndarray], percentiles: list[float] | None = None) -> list[np.ndarray]:
        """
        Nyúl histogram normalization for multiple images.

        Standardizes intensity ranges across a cohort of images.
        """
        if percentiles is None:
            percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]

        # Calculate landmarks for each image
        landmarks = []
        for img in images:
            brain_voxels = img[img > 0]  # Assume background is 0
            img_landmarks = np.percentile(brain_voxels, percentiles)
            landmarks.append(img_landmarks)

        # Calculate average landmarks
        avg_landmarks = np.mean(landmarks, axis=0)

        # Normalize each image
        normalized_images = []
        for i, img in enumerate(images):
            # Piecewise linear mapping
            normalized = np.interp(img, landmarks[i], avg_landmarks)
            normalized_images.append(normalized)

        return normalized_images


class BiasFieldCorrection:
    """N4 bias field correction implementation."""

    def __init__(self, max_iterations: int = 50, convergence_threshold: float = 0.001):
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold

    def correct_bias_field(self, image: np.ndarray, mask: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        """
        Simplified bias field correction using polynomial fitting.

        In practice, would use proper N4ITK implementation.
        """
        if mask is None:
            mask = image > 0

        # Create coordinate grids
        z, y, x = np.mgrid[0 : image.shape[0], 0 : image.shape[1], 0 : image.shape[2]]
        coords = np.column_stack([z.ravel(), y.ravel(), x.ravel()])

        # Fit polynomial to log intensities
        log_image = np.log(image + 1e-8)
        mask_flat = mask.ravel()

        # Use polynomial features (simplified)
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import PolynomialFeatures

        poly = PolynomialFeatures(degree=3)
        coords_poly = poly.fit_transform(coords[mask_flat])

        # Fit bias field
        ridge = Ridge(alpha=1.0)
        ridge.fit(coords_poly, log_image.ravel()[mask_flat])

        # Predict bias field for entire image
        coords_poly_full = poly.transform(coords)
        bias_field = ridge.predict(coords_poly_full).reshape(image.shape)

        # Correct image
        corrected = image / (np.exp(bias_field) + 1e-8)

        return corrected, np.exp(bias_field)


class MedicalImageAugmentation:
    """Medical image-specific augmentation techniques."""

    def __init__(self, config: PreprocessingConfig):
        self.config = config

    def elastic_deformation(self, image: np.ndarray, alpha: float = 2000, sigma: float = 50) -> np.ndarray:
        """Apply elastic deformation to simulate anatomical variations."""
        shape = image.shape

        # Generate random displacement fields
        dx = np.random.randn(*shape) * sigma
        dy = np.random.randn(*shape) * sigma
        dz = np.random.randn(*shape) * sigma

        # Smooth displacement fields
        dx = ndimage.gaussian_filter(dx, sigma, mode="constant", cval=0) * alpha
        dy = ndimage.gaussian_filter(dy, sigma, mode="constant", cval=0) * alpha
        dz = ndimage.gaussian_filter(dz, sigma, mode="constant", cval=0) * alpha

        # Create coordinate grids
        z, y, x = np.mgrid[0 : shape[0], 0 : shape[1], 0 : shape[2]]
        indices = z + dz, y + dy, x + dx

        # Apply deformation
        deformed = ndimage.map_coordinates(image, indices, order=1, mode="reflect")
        return deformed

    def add_noise(self, image: np.ndarray, noise_type: str = "gaussian", intensity: float = 0.1) -> np.ndarray:
        """Add realistic noise to medical images."""
        if noise_type == "gaussian":
            noise = np.random.normal(0, intensity * np.std(image), image.shape)
            return image + noise

        elif noise_type == "rician":
            # Rician noise (common in MRI)
            sigma = intensity * np.std(image)
            real_part = image + np.random.normal(0, sigma, image.shape)
            imag_part = np.random.normal(0, sigma, image.shape)
            return np.sqrt(real_part**2 + imag_part**2)

        elif noise_type == "poisson":
            # Poisson noise (common in CT)
            # Scale image to appropriate range for Poisson
            scaled = image / np.max(image) * 100
            noisy = np.random.poisson(scaled)
            return noisy / 100 * np.max(image)

        return image

    def simulate_motion_artifacts(self, image: np.ndarray, severity: float = 0.1) -> np.ndarray:
        """Simulate motion artifacts in medical images."""
        # Simple motion blur simulation
        kernel_size = int(severity * 10) + 1
        if kernel_size % 2 == 0:
            kernel_size += 1

        # Random motion direction
        angle = np.random.uniform(0, 360)
        kernel = self._create_motion_blur_kernel(kernel_size, angle)

        # Apply convolution to each slice
        blurred = np.zeros_like(image)
        for i in range(image.shape[0]):
            blurred[i] = cv2.filter2D(image[i], -1, kernel)

        return blurred

    def _create_motion_blur_kernel(self, size: int, angle: float) -> np.ndarray:
        """Create motion blur kernel."""
        kernel = np.zeros((size, size))
        center = size // 2

        # Calculate line endpoints
        angle_rad = np.radians(angle)
        dx = int(center * np.cos(angle_rad))
        dy = int(center * np.sin(angle_rad))

        # Draw line in kernel
        cv2.line(kernel, (center - dx, center - dy), (center + dx, center + dy), 1, 1)

        # Normalize
        kernel = kernel / np.sum(kernel)
        return kernel

    def intensity_shift(self, image: np.ndarray, shift_range: float = 0.2) -> np.ndarray:
        """Apply random intensity shifts."""
        shift_factor = np.random.uniform(1 - shift_range, 1 + shift_range)
        return image * shift_factor

    def gamma_correction(self, image: np.ndarray, gamma_range: tuple[float, float] = (0.8, 1.2)) -> np.ndarray:
        """Apply random gamma correction."""
        gamma = np.random.uniform(*gamma_range)
        # Normalize to [0, 1] for gamma correction
        img_norm = (image - np.min(image)) / (np.max(image) - np.min(image) + 1e-8)
        corrected = np.power(img_norm, gamma)
        # Scale back to original range
        return corrected * (np.max(image) - np.min(image)) + np.min(image)


class AdvancedPreprocessingPipeline:
    """Complete preprocessing pipeline for multimodal brain tumor segmentation."""

    def __init__(self, config: PreprocessingConfig = None):
        self.config = config or PreprocessingConfig()
        self.registration = MutualInformationRegistration()
        self.normalization = AdvancedNormalization()
        self.bias_correction = BiasFieldCorrection()
        self.augmentation = MedicalImageAugmentation(self.config)

        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def preprocess_pair(
        self,
        ct_path: str | Path,
        mri_path: str | Path,
        mask_path: str | Path | None = None,
        apply_augmentation: bool = False,
    ) -> dict[str, np.ndarray]:
        """
        Complete preprocessing of CT-MRI pair.

        Args:
            ct_path: Path to CT image
            mri_path: Path to MRI image
            mask_path: Optional path to segmentation mask
            apply_augmentation: Whether to apply data augmentation

        Returns:
            Dictionary containing processed images and metadata
        """
        self.logger.info(f"Processing pair: {ct_path}, {mri_path}")

        # Load images
        ct_img = self._load_medical_image(ct_path)
        mri_img = self._load_medical_image(mri_path)

        if mask_path:
            mask = self._load_medical_image(mask_path)
        else:
            mask = None

        # Quality assessment
        ct_quality = self._assess_image_quality(ct_img)
        mri_quality = self._assess_image_quality(mri_img)

        self.logger.info(f"Image quality - CT: {ct_quality:.3f}, MRI: {mri_quality:.3f}")

        # Bias field correction
        if self.config.bias_field_correction:
            ct_img, _ = self.bias_correction.correct_bias_field(ct_img)
            mri_img, _ = self.bias_correction.correct_bias_field(mri_img)

        # Registration (register MRI to CT space)
        self.logger.info("Performing image registration...")
        mri_registered, transform_params = self.registration.register(ct_img, mri_img)

        # Apply same transformation to mask if provided
        if mask is not None:
            mask_registered = self.registration._apply_affine_transform(mask, transform_params)
        else:
            mask_registered = None

        # Intensity normalization
        if self.config.normalization_method == "zscore":
            ct_normalized = self.normalization.zscore_normalization(ct_img, mask)
            mri_normalized = self.normalization.zscore_normalization(mri_registered, mask_registered)
        elif self.config.normalization_method == "zscore_robust":
            ct_normalized = self.normalization.robust_zscore_normalization(ct_img, mask)
            mri_normalized = self.normalization.robust_zscore_normalization(mri_registered, mask_registered)
        elif self.config.normalization_method == "histogram_matching":
            ct_normalized = ct_img
            mri_normalized = self.normalization.histogram_matching(mri_registered, ct_img, mask)
        else:
            ct_normalized = ct_img
            mri_normalized = mri_registered

        # Resample to target spacing and size
        ct_resampled = self._resample_image(ct_normalized, self.config.target_size)
        mri_resampled = self._resample_image(mri_normalized, self.config.target_size)

        if mask_registered is not None:
            mask_resampled = self._resample_image(mask_registered, self.config.target_size, interpolation="nearest")
        else:
            mask_resampled = None

        # Data augmentation
        if apply_augmentation and np.random.random() < self.config.augmentation_probability:
            ct_resampled, mri_resampled, mask_resampled = self._apply_augmentation(
                ct_resampled, mri_resampled, mask_resampled
            )

        result = {
            "ct": ct_resampled,
            "mri": mri_resampled,
            "mask": mask_resampled,
            "transform_params": transform_params,
            "quality_scores": {"ct": ct_quality, "mri": mri_quality},
        }

        return result

    def _load_medical_image(self, path: str | Path) -> np.ndarray:
        """Load medical image (NIfTI format)."""
        path = Path(path)

        if path.suffix in [".nii", ".nii.gz"]:
            img = nib.load(str(path))
            return img.get_fdata()
        else:
            # Fallback to OpenCV for other formats
            img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise ValueError(f"Could not load image: {path}")
            return img.astype(np.float32)

    def _assess_image_quality(self, image: np.ndarray) -> float:
        """Assess image quality using various metrics."""
        # Signal-to-noise ratio estimation
        signal = np.mean(image[image > 0])
        noise = np.std(image[image > 0])
        snr = signal / (noise + 1e-8)

        # Contrast measure
        contrast = np.std(image)

        # Sharpness measure (gradient magnitude)
        grad_x = np.gradient(image, axis=0)
        grad_y = np.gradient(image, axis=1)
        if image.ndim == 3:
            grad_z = np.gradient(image, axis=2)
            sharpness = np.mean(np.sqrt(grad_x**2 + grad_y**2 + grad_z**2))
        else:
            sharpness = np.mean(np.sqrt(grad_x**2 + grad_y**2))

        # Combine metrics (normalize and weight)
        quality_score = 0.4 * min(snr / 10.0, 1.0) + 0.3 * min(contrast / 100.0, 1.0) + 0.3 * min(sharpness / 50.0, 1.0)

        return quality_score

    def _resample_image(
        self, image: np.ndarray, target_size: tuple[int, int, int], interpolation: str = "linear"
    ) -> np.ndarray:
        """Resample image to target size."""
        if image.ndim == 2:
            # 2D image
            if interpolation == "nearest":
                interpolation_cv = cv2.INTER_NEAREST
            else:
                interpolation_cv = cv2.INTER_LINEAR

            resampled = cv2.resize(image, target_size[:2], interpolation=interpolation_cv)
            return resampled

        elif image.ndim == 3:
            # 3D image - resample each dimension
            zoom_factors = [target_size[i] / image.shape[i] for i in range(3)]

            if interpolation == "nearest":
                order = 0
            else:
                order = 1

            resampled = ndimage.zoom(image, zoom_factors, order=order)
            return resampled

        else:
            raise ValueError(f"Unsupported image dimensions: {image.ndim}")

    def _apply_augmentation(
        self, ct: np.ndarray, mri: np.ndarray, mask: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        """Apply random augmentations to image pair."""
        augmentation_type = np.random.choice(["elastic", "noise", "intensity_shift", "gamma", "motion"])

        if augmentation_type == "elastic":
            ct_aug = self.augmentation.elastic_deformation(ct)
            mri_aug = self.augmentation.elastic_deformation(mri)
            mask_aug = self.augmentation.elastic_deformation(mask) if mask is not None else None

        elif augmentation_type == "noise":
            noise_type = np.random.choice(["gaussian", "rician"])
            ct_aug = self.augmentation.add_noise(ct, noise_type)
            mri_aug = self.augmentation.add_noise(mri, noise_type)
            mask_aug = mask  # Don't add noise to mask

        elif augmentation_type == "intensity_shift":
            ct_aug = self.augmentation.intensity_shift(ct)
            mri_aug = self.augmentation.intensity_shift(mri)
            mask_aug = mask

        elif augmentation_type == "gamma":
            ct_aug = self.augmentation.gamma_correction(ct)
            mri_aug = self.augmentation.gamma_correction(mri)
            mask_aug = mask

        elif augmentation_type == "motion":
            ct_aug = self.augmentation.simulate_motion_artifacts(ct)
            mri_aug = self.augmentation.simulate_motion_artifacts(mri)
            mask_aug = mask

        else:
            ct_aug, mri_aug, mask_aug = ct, mri, mask

        return ct_aug, mri_aug, mask_aug

    def batch_preprocess(
        self, data_pairs: list[tuple[str, str, str | None]], output_dir: str | Path, n_jobs: int = 1
    ) -> list[dict]:
        """
        Batch preprocessing of multiple image pairs.

        Args:
            data_pairs: List of (ct_path, mri_path, mask_path) tuples
            output_dir: Directory to save processed data
            n_jobs: Number of parallel jobs

        Returns:
            List of processing results
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        results = []

        for i, (ct_path, mri_path, mask_path) in enumerate(data_pairs):
            try:
                result = self.preprocess_pair(ct_path, mri_path, mask_path)

                # Save processed data
                case_id = f"case_{i:04d}"
                case_dir = output_dir / case_id
                case_dir.mkdir(exist_ok=True)

                np.save(case_dir / "ct.npy", result["ct"])
                np.save(case_dir / "mri.npy", result["mri"])
                if result["mask"] is not None:
                    np.save(case_dir / "mask.npy", result["mask"])

                # Save metadata
                metadata = {
                    "transform_params": result["transform_params"].tolist(),
                    "quality_scores": result["quality_scores"],
                    "original_paths": {
                        "ct": str(ct_path),
                        "mri": str(mri_path),
                        "mask": str(mask_path) if mask_path else None,
                    },
                }

                import json

                with open(case_dir / "metadata.json", "w") as f:
                    json.dump(metadata, f, indent=2)

                results.append({"case_id": case_id, "status": "success", **result})

            except Exception as e:
                self.logger.error(f"Failed to process pair {i}: {e}")
                results.append({"case_id": f"case_{i:04d}", "status": "failed", "error": str(e)})

        return results


def main():
    """Demonstration of advanced preprocessing pipeline."""
    config = PreprocessingConfig(
        target_size=(128, 128, 128),
        bias_field_correction=True,
        normalization_method="zscore_robust",
        augmentation_probability=0.3,
    )

    AdvancedPreprocessingPipeline(config)

    # Example usage (with dummy paths)

    # Batch processing
    # results = pipeline.batch_preprocess(data_pairs, "/path/to/output")

    print("Advanced preprocessing pipeline initialized successfully!")
    print(f"Configuration: {config}")


if __name__ == "__main__":
    main()
