import json
import os
import platform
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class RunMetadata:
    seed: int
    mode: str
    output_dir: str
    git_commit: Optional[str]
    python_version: str
    platform: str
    cuda: Optional[str]
    torch: Optional[str]


def _get_git_commit() -> Optional[str]:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode("utf-8")
            .strip()
        )
    except Exception:
        return None


def _get_torch_env() -> Dict[str, Optional[str]]:
    try:
        import torch  # type: ignore

        cuda = torch.version.cuda if hasattr(torch, "version") else None
        return {"torch": torch.__version__, "cuda": cuda}
    except Exception:
        return {"torch": None, "cuda": None}


def write_metadata(output_dir: Path, seed: int, mode: str) -> Path:
    info = _get_torch_env()
    md = RunMetadata(
        seed=seed,
        mode=mode,
        output_dir=str(output_dir),
        git_commit=_get_git_commit(),
        python_version=platform.python_version(),
        platform=platform.platform(),
        cuda=info["cuda"],
        torch=info["torch"],
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "metadata.json"
    path.write_text(json.dumps(asdict(md), indent=2))
    return path


