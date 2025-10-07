from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional


class MLflowTracker:
    """Lightweight MLflow wrapper. Fails silent if mlflow is not installed.

    Usage:
        tracker = MLflowTracker(enabled=True, experiment_name="exp", tracking_uri="file:./mlruns")
        with tracker.start_run(run_name="demo"):
            tracker.log_params({"seed": 42, "mode": "demo"})
            tracker.log_metrics({"dice": 0.85}, step=1)
            tracker.log_artifact("enhanced_framework_results/summary_report.txt")
    """

    def __init__(
        self,
        enabled: bool = False,
        experiment_name: Optional[str] = None,
        tracking_uri: Optional[str] = None,
    ) -> None:
        self.enabled = enabled
        self._mlflow = None
        self._active_run = None

        if not self.enabled:
            return

        try:
            import mlflow  # type: ignore

            self._mlflow = mlflow
            if tracking_uri:
                mlflow.set_tracking_uri(tracking_uri)
            if experiment_name:
                mlflow.set_experiment(experiment_name)
        except Exception:
            # Disable if mlflow unavailable or misconfigured
            self.enabled = False
            self._mlflow = None

    def start_run(self, run_name: Optional[str] = None):  # type: ignore[override]
        if not self.enabled or self._mlflow is None:
            return _NullContext()
        return self._mlflow.start_run(run_name=run_name)

    def log_params(self, params: Dict[str, Any]) -> None:
        if not self.enabled or self._mlflow is None:
            return
        # mlflow expects flat dict with str, int, float, bool; stringify others
        safe_params: Dict[str, Any] = {}
        for k, v in params.items():
            if isinstance(v, (str, int, float, bool)) or v is None:
                safe_params[k] = v
            else:
                try:
                    safe_params[k] = json.dumps(v, ensure_ascii=False)
                except Exception:
                    safe_params[k] = str(v)
        try:
            self._mlflow.log_params(safe_params)
        except Exception:
            pass

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        if not self.enabled or self._mlflow is None:
            return
        try:
            self._mlflow.log_metrics(metrics, step=step)
        except Exception:
            pass

    def log_artifact(self, path: str) -> None:
        if not self.enabled or self._mlflow is None:
            return
        try:
            if os.path.isfile(path):
                self._mlflow.log_artifact(path)
        except Exception:
            pass

    def log_dict(self, data: Dict[str, Any], artifact_file: str) -> None:
        if not self.enabled or self._mlflow is None:
            return
        try:
            self._mlflow.log_dict(data, artifact_file)
        except Exception:
            # Fallback: write locally and log as artifact
            try:
                tmp = Path(artifact_file)
                tmp.parent.mkdir(parents=True, exist_ok=True)
                tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False))
                self.log_artifact(str(tmp))
            except Exception:
                pass


class _NullContext:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


