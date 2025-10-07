## Reproducibility

This repository enforces reproducibility through:
- Global seeding via `--seed` and `tools/seed_utils.py` (NumPy/Python/Torch/CuDNN determinism)
- Metadata logging (`metadata.json` with env, commit, seed)
- MLflow tracking (params/metrics/artifacts)
- CI-run CPU demo tests for quick regression

### Recommended run
```bash
python run_enhanced_framework.py --mode demo --seed 42 --mlflow --mlflow_uri file:./mlruns
```

### Determinism notes
- CuDNN `deterministic=True, benchmark=False` trades speed for determinism.
- For fair comparison, use same seeds and fixed validation splits.


