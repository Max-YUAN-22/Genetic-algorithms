from __future__ import annotations

from pathlib import Path

from run_enhanced_framework import EnhancedFrameworkRunner
from tools.seed_utils import set_global_seed
from tools.stats import cohen_d, mean_std, ttest_independent


def run_demo_seeds(output_dir: Path, seeds: list[int]) -> dict[str, float]:
    scores: list[float] = []
    for s in seeds:
        set_global_seed(s)
        runner = EnhancedFrameworkRunner(str(output_dir / f"seed_{s}"))
        res = runner.run_demo()
        med = res.get("medical_evaluation", {}) if isinstance(res, dict) else {}
        mean_dice = med.get("clinical_summary", {}).get("mean_dice") if isinstance(med, dict) else None
        if isinstance(mean_dice, (int, float)):
            scores.append(float(mean_dice))
    summary = mean_std(scores)
    return {"mean": summary.mean, "std": summary.std, "n": float(summary.n)}


def compare_runs(x_scores: list[float], y_scores: list[float]) -> dict[str, float]:
    t, dof = ttest_independent(x_scores, y_scores)
    d = cohen_d(x_scores, y_scores)
    return {"t_stat": t, "dof": dof, "cohen_d": d}
