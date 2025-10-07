from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple


@dataclass
class Summary:
    mean: float
    std: float
    n: int


def mean_std(values: Iterable[float]) -> Summary:
    xs = list(float(v) for v in values)
    n = len(xs)
    if n == 0:
        return Summary(mean=float('nan'), std=float('nan'), n=0)
    m = sum(xs) / n
    var = sum((x - m) ** 2 for x in xs) / (n - 1) if n > 1 else 0.0
    return Summary(mean=m, std=var ** 0.5, n=n)


def cohen_d(x: Iterable[float], y: Iterable[float]) -> float:
    x, y = list(map(float, x)), list(map(float, y))
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return float('nan')
    mx, my = sum(x) / nx, sum(y) / ny
    vx = sum((v - mx) ** 2 for v in x) / (nx - 1)
    vy = sum((v - my) ** 2 for v in y) / (ny - 1)
    sp = (((nx - 1) * vx + (ny - 1) * vy) / (nx + ny - 2)) ** 0.5
    return (mx - my) / sp if sp > 0 else float('inf')


def ttest_independent(x: Iterable[float], y: Iterable[float]) -> Tuple[float, float]:
    """Simple Welch's t-test (approximate). Returns t-stat and dof-only p placeholder.

    Note: To avoid heavy dependencies, we return (t, dof) for reporting,
    and recommend using scipy on reviewers' side for exact p-values.
    """
    x, y = list(map(float, x)), list(map(float, y))
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return float('nan'), float('nan')
    mx, my = sum(x) / nx, sum(y) / ny
    vx = sum((v - mx) ** 2 for v in x) / (nx - 1)
    vy = sum((v - my) ** 2 for v in y) / (ny - 1)
    num = mx - my
    den = (vx / nx + vy / ny) ** 0.5
    t = num / den if den > 0 else float('inf')
    # Welch–Satterthwaite DOF approximation
    dof_num = (vx / nx + vy / ny) ** 2
    dof_den = (vx ** 2) / (nx ** 2 * (nx - 1)) + (vy ** 2) / (ny ** 2 * (ny - 1))
    dof = dof_num / dof_den if dof_den > 0 else float('inf')
    return t, dof


