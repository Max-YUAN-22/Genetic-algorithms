#!/usr/bin/env python3
"""
Aggregate results from evaluation outputs and export WT/TC/ET metrics tables and LaTeX.

This script scans directories like brats_enhanced_results/, evaluation_results/, sota_validation_results/ and
consolidates per-case and aggregate metrics into CSV and LaTeX tables suitable for Q2投稿。
"""

import csv
import json
from pathlib import Path
from typing import Any


def _find_json_files(root: Path) -> list[Path]:
    files = []
    for p in root.rglob("*.json"):
        if p.name.endswith(
            (
                "clinical_report.json",
                "sota_validation_report.json",
                "detailed_results.json",
                "real_eval_only_results.json",
                "real_training_results.json",
            )
        ):
            files.append(p)
    return files


def _extract_brats_summary(report: dict[str, Any]) -> dict[str, float]:
    summary = {}
    # Try clinical report structure
    if "evaluation_summary" in report:
        es = report["evaluation_summary"]
        summary["mean_dice"] = float(es.get("mean_dice_coefficient", 0.0))
    # Try sota report
    if "model_performance" in report:
        # pick our model if present
        if "ours_multimodal_ga" in report["model_performance"]:
            perf = report["model_performance"]["ours_multimodal_ga"]
            summary["ours_mean_dice"] = float(perf.get("dice_coefficient", 0.0))
    # Try real eval/training results structure
    for key in ["detailed_wt_tc_et", "detailed_results", "batch_statistics"]:
        if key in report and isinstance(report[key], dict):
            s = report[key]
            # generic mean_dice
            if "mean_dice" in s:
                summary["mean_dice"] = float(s.get("mean_dice", 0.0))
            # BraTS region metrics
            for m in [
                "WT_Dice_mean",
                "TC_Dice_mean",
                "ET_Dice_mean",
                "WT_Hausdorff95_mean",
                "TC_Hausdorff95_mean",
                "ET_Hausdorff95_mean",
            ]:
                if m in s and isinstance(s[m], (int, float)):
                    summary[m] = float(s[m])
    # Optionally pull WT/TC/ET if available in stats
    for key in [
        "WT_Dice_mean",
        "TC_Dice_mean",
        "ET_Dice_mean",
        "WT_Hausdorff95_mean",
        "TC_Hausdorff95_mean",
        "ET_Hausdorff95_mean",
    ]:
        if key in report:
            summary[key] = float(report[key])
    return summary


def aggregate_results(roots: list[str], out_dir: str = "publication_results") -> dict[str, Any]:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    aggregated: dict[str, Any] = {"files": [], "summaries": []}
    for root in roots:
        r = Path(root)
        if not r.exists():
            continue
        for jf in _find_json_files(r):
            try:
                data = json.loads(jf.read_text())
                summary = _extract_brats_summary(data)
                aggregated["files"].append(str(jf))
                aggregated["summaries"].append(summary)
            except Exception:
                continue

    # Export CSV
    csv_path = out_path / "brats_wt_tc_et_summary.csv"
    # Ensure stable, useful columns
    preferred_cols = [
        "mean_dice",
        "ours_mean_dice",
        "WT_Dice_mean",
        "TC_Dice_mean",
        "ET_Dice_mean",
        "WT_Hausdorff95_mean",
        "TC_Hausdorff95_mean",
        "ET_Hausdorff95_mean",
    ]
    existing = sorted({k for s in aggregated["summaries"] for k in s.keys()})
    fieldnames = [c for c in preferred_cols if c in existing] + [c for c in existing if c not in preferred_cols]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for s in aggregated["summaries"]:
            writer.writerow(s)

    # Export LaTeX skeleton
    tex_path = out_path / "table_brats_regions.tex"
    latex = [
        "\\begin{table}[h]",
        "\\centering",
        "\\caption{BraTS WT/TC/ET performance (Dice and HD95).}",
        "\\begin{tabular}{lcccccc}",
        "\\hline",
        "Method & WT Dice$\\uparrow$ & TC Dice$\\uparrow$ & ET Dice$\\uparrow$ & WT HD95$\\downarrow$ & TC HD95$\\downarrow$ & ET HD95$\\downarrow$ \\\\",
        "\\hline",
        "Ours & - & - & - & - & - & - \\\\",
        "nnU-Net & - & - & - & - & - & - \\\\",
        "\\hline",
        "\\end{tabular}",
        "\\label{tab:brats_regions}",
        "\\end{table}",
    ]
    tex_path.write_text("\n".join(latex))

    out_json = out_path / "aggregated_publication_results.json"
    out_json.write_text(json.dumps(aggregated, indent=2))

    return {"csv": str(csv_path), "tex": str(tex_path), "json": str(out_json)}


def main():
    roots = [
        "brats_enhanced_results",
        "enhanced_framework_results",
        "sota_validation_results",
        "real_training_results",
    ]
    outputs = aggregate_results(roots)
    print("Aggregated outputs:", outputs)


if __name__ == "__main__":
    main()
