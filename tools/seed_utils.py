import os
import random
from typing import Optional


def set_global_seed(seed: Optional[int] = None) -> int:
    """
    Set global RNG seeds for reproducibility across Python, NumPy, and PyTorch if available.

    Returns the effective seed used to initialize all RNGs.
    """
    if seed is None:
        env_seed = os.getenv("SEED")
        seed = int(env_seed) if env_seed is not None and env_seed.isdigit() else 42

    try:
        import numpy as np  # type: ignore

        np.random.seed(seed)
    except Exception:
        pass

    random.seed(seed)

    try:
        import torch  # type: ignore

        torch.manual_seed(seed)
        if torch.cuda.is_available():  # type: ignore[attr-defined]
            torch.cuda.manual_seed_all(seed)  # type: ignore[attr-defined]
        # Determinism trade-offs: safer defaults for medical imaging experiments
        try:
            import torch.backends.cudnn as cudnn  # type: ignore

            cudnn.deterministic = True
            cudnn.benchmark = False
        except Exception:
            pass
    except Exception:
        pass

    os.environ["PYTHONHASHSEED"] = str(seed)
    return seed
