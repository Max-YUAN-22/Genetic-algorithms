## Methods

We propose a multimodal segmentation framework that couples a YOLO-style encoder–decoder with a cross-modal attention (CMA) fusion module and a multi-objective genetic tuner. Two modality-specific streams (CT, MRI) extract features in parallel. At each decoder scale, CMA exchanges information via query–key–value attention across streams, followed by residual gating and LayerNorm for numerical stability. The tuner searches over architectural and training hyperparameters—depth/width, kernel sizes, CMA on/off, augmentation policies, and postprocessing thresholds—under constraints targeting a Pareto front on accuracy (Dice), efficiency (FLOPs and latency), and predictive uncertainty (variance under MC Dropout/TTA).

Uncertainty is estimated by enabling dropout at inference or by test-time augmentation, producing voxel-wise variance maps. Postprocessing includes class-wise largest-component retention, minimum component size filtering, and morphological closing to enforce clinical plausibility. The framework is engineered for reproducibility through global seeding, deterministic cuDNN settings, MLflow logging, and environment metadata capture; a unified CLI ensures consistent runs across machines.

## Experiments

We evaluate on BRaTS-style segmentation tasks and synthetic validations. Data undergo bias-field correction and robust normalization, with fixed train/validation splits and seeds. Baselines include U-Net, Attention U-Net, nnU-Net, and YOLO-based segmentation. Metrics cover Dice (WT/TC/ET), Jaccard, Hausdorff95, sensitivity/specificity, and calibration (ECE). We report mean±std over ≥5 independent runs; statistical significance is assessed by Welch’s t-test alongside effect size (Cohen’s d). Runtime and memory are measured on CPU for portability and on a representative GPU for throughput. Qualitative analyses visualize uncertainty heatmaps and typical/edge clinical cases to substantiate decision support.


