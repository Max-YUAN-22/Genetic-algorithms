import json
from pathlib import Path


def test_demo_runs(tmp_path: Path):
    # Run demo via module to avoid needing CLI tools
    from run_enhanced_framework import EnhancedFrameworkRunner

    out = tmp_path / "demo_out"
    runner = EnhancedFrameworkRunner(str(out))
    results = runner.run_demo()

    # Results shape
    assert isinstance(results, dict)
    assert results.get("preprocessing", {}).get("status") == "completed"
    assert results.get("genetic_optimization", {}).get("status") == "completed"
    assert results.get("medical_evaluation", {}).get("status") == "completed"
    assert results.get("sota_comparison", {}).get("status") == "completed"

    # Artifacts exist
    demo_json = out / "demo_results.json"
    assert demo_json.exists()
    data = json.loads(demo_json.read_text())
    assert "medical_evaluation" in data
