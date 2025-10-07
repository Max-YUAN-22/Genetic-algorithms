## Experiments

### Ablation (demo-scale)
```bash
python -m tools.cli demo --seed 0
python -m tools.cli demo --seed 1
python -m tools.cli demo --seed 2
```
Aggregate MLflow runs to report mean±std of key metrics (e.g., mean Dice).

### SOTA comparison (synthetic)
```bash
python -m tools.cli validate --seed 42
```
Produces per-model Dice summary in `enhanced_framework_results/sota_demo/`.

### Real data (template)
- Prepare dataset using `examples/ct_mri_fusion.py` and `examples/ct_mri_seg.yaml`
- Run training
```bash
python -m tools.cli train --data-path /abs/path/to/dataset --seed 42
```


