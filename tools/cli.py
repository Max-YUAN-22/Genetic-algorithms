from typing import Optional

import typer

from run_enhanced_framework import EnhancedFrameworkRunner
from tools.mlflow_tracking import MLflowTracker
from tools.seed_utils import set_global_seed

app = typer.Typer(help="Unified CLI for the enhanced multimodal brain tumor segmentation framework")


def build_tracker(enable: bool, experiment: str, uri: str) -> MLflowTracker:
    return MLflowTracker(enabled=enable, experiment_name=experiment, tracking_uri=uri)


@app.command()
def demo(
    output_dir: str = typer.Option("enhanced_framework_results", help="Output directory"),
    seed: int = typer.Option(42, help="Global random seed"),
    mlflow: bool = typer.Option(False, help="Enable MLflow tracking"),
    mlflow_experiment: str = typer.Option("enhanced_framework", help="MLflow experiment name"),
    mlflow_uri: str = typer.Option("file:./mlruns", help="MLflow tracking URI"),
):
    set_global_seed(seed)
    runner = EnhancedFrameworkRunner(output_dir)
    tracker = build_tracker(mlflow, mlflow_experiment, mlflow_uri)
    with tracker.start_run(run_name="demo"):
        tracker.log_params({"mode": "demo", "seed": seed, "output_dir": output_dir})
        results = runner.run_demo()
        med = results.get("medical_evaluation", {}) if isinstance(results, dict) else {}
        mean_dice = med.get("clinical_summary", {}).get("mean_dice") if isinstance(med, dict) else None
        if isinstance(mean_dice, (int, float)):
            tracker.log_metrics({"mean_dice": float(mean_dice)})
        # persist report
        summary = runner.generate_summary_report(results)
        report_path = runner.output_dir / "summary_report.txt"
        report_path.write_text(summary)
        tracker.log_artifact(str(report_path))


@app.command()
def train(
    data_path: Optional[str] = typer.Option(None, help="Path to training data"),
    output_dir: str = typer.Option("enhanced_framework_results", help="Output directory"),
    seed: int = typer.Option(42, help="Global random seed"),
    mlflow: bool = typer.Option(False, help="Enable MLflow tracking"),
    mlflow_experiment: str = typer.Option("enhanced_framework", help="MLflow experiment name"),
    mlflow_uri: str = typer.Option("file:./mlruns", help="MLflow tracking URI"),
):
    set_global_seed(seed)
    runner = EnhancedFrameworkRunner(output_dir)
    tracker = build_tracker(mlflow, mlflow_experiment, mlflow_uri)
    with tracker.start_run(run_name="train"):
        tracker.log_params({"mode": "train", "seed": seed, "output_dir": output_dir})
        results = runner.run_training(data_path)
        summary = runner.generate_summary_report(results)
        report_path = runner.output_dir / "summary_report.txt"
        report_path.write_text(summary)
        tracker.log_artifact(str(report_path))


@app.command()
def validate(
    test_data: Optional[str] = typer.Option(None, help="Path to test data"),
    output_dir: str = typer.Option("enhanced_framework_results", help="Output directory"),
    seed: int = typer.Option(42, help="Global random seed"),
    mlflow: bool = typer.Option(False, help="Enable MLflow tracking"),
    mlflow_experiment: str = typer.Option("enhanced_framework", help="MLflow experiment name"),
    mlflow_uri: str = typer.Option("file:./mlruns", help="MLflow tracking URI"),
):
    set_global_seed(seed)
    runner = EnhancedFrameworkRunner(output_dir)
    tracker = build_tracker(mlflow, mlflow_experiment, mlflow_uri)
    with tracker.start_run(run_name="validate"):
        tracker.log_params({"mode": "validate", "seed": seed, "output_dir": output_dir})
        results = runner.run_validation(test_data_path=test_data)
        summary = runner.generate_summary_report(results)
        report_path = runner.output_dir / "summary_report.txt"
        report_path.write_text(summary)
        tracker.log_artifact(str(report_path))


@app.command()
def full(
    data_dir: Optional[str] = typer.Option(None, help="Root data directory"),
    output_dir: str = typer.Option("enhanced_framework_results", help="Output directory"),
    seed: int = typer.Option(42, help="Global random seed"),
    mlflow: bool = typer.Option(False, help="Enable MLflow tracking"),
    mlflow_experiment: str = typer.Option("enhanced_framework", help="MLflow experiment name"),
    mlflow_uri: str = typer.Option("file:./mlruns", help="MLflow tracking URI"),
):
    set_global_seed(seed)
    runner = EnhancedFrameworkRunner(output_dir)
    tracker = build_tracker(mlflow, mlflow_experiment, mlflow_uri)
    with tracker.start_run(run_name="full"):
        tracker.log_params({"mode": "full", "seed": seed, "output_dir": output_dir})
        results = runner.run_full_pipeline(data_dir)
        summary = runner.generate_summary_report(results)
        report_path = runner.output_dir / "summary_report.txt"
        report_path.write_text(summary)
        tracker.log_artifact(str(report_path))


if __name__ == "__main__":
    app()
