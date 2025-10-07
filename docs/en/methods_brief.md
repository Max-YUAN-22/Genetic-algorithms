## Methods Brief

### Architecture
- YOLO-style encoder-decoder with segmentation head; dual-stream CT/MRI backbones.
- Cross-Modal Attention (CMA): per-scale Q/K/V across streams with residual gating and LayerNorm; temperature scaling for stable fusion.

### Genetic Tuning
- Multi-objective: maximize Dice; minimize FLOPs/latency; minimize uncertainty (variance under MC Dropout/TTA).
- Gene space: width/depth, kernel sizes, attention on/off, postprocess thresholds, augmentation policies.
- Operators: tournament selection, simulated binary crossover, Gaussian mutation; elitism and early stopping.

### Uncertainty & Postprocessing
- MC Dropout or TTA to estimate voxel-wise variance; calibrated thresholding.
- Morphological cleanup (keep largest per class, min component size, closing radius).

### Training & Eval
- Robust normalization; fixed seeds; reporting mean±std over ≥5 runs; Welch t-test and Cohen’s d for significance and effect size.


