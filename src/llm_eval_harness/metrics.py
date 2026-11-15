"""Agreement, stability and bootstrap statistics. Pure Python, no numpy."""

from __future__ import annotations

import math
import random
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

Weights = Literal["linear", "quadratic"] | None


def _paired(a: Sequence[float], b: Sequence[float]) -> None:
    if len(a) != len(b):
        raise ValueError(f"length mismatch: {len(a)} vs {len(b)}")
    if not a:
        raise ValueError("need at least one pair")


def exact_agreement(a: Sequence[int], b: Sequence[int]) -> float:
    _paired(a, b)
    return sum(x == y for x, y in zip(a, b, strict=True)) / len(a)


def within_agreement(a: Sequence[int], b: Sequence[int], tolerance: int = 1) -> float:
    _paired(a, b)
    return sum(abs(x - y) <= tolerance for x, y in zip(a, b, strict=True)) / len(a)


def mean_abs_error(a: Sequence[float], b: Sequence[float]) -> float:
    _paired(a, b)
    return sum(abs(x - y) for x, y in zip(a, b, strict=True)) / len(a)


def mean_signed_difference(a: Sequence[float], b: Sequence[float]) -> float:
    """Mean of (a - b). With a = judge and b = human, a positive value means a lenient judge."""
    _paired(a, b)
    return sum(x - y for x, y in zip(a, b, strict=True)) / len(a)


def cohens_kappa(
    a: Sequence[int],
    b: Sequence[int],
    *,
    labels: Sequence[int] | None = None,
    weights: Weights = None,
) -> float:
    """Cohen's kappa between two raters on an ordered integer scale.

    `weights` None gives plain kappa, "linear" and "quadratic" penalise larger gaps more.
    Returns NaN when chance agreement is perfect (for example both raters always give the
    same single label), because the statistic is undefined there.
    """
    _paired(a, b)
    label_list = sorted(set(labels) if labels is not None else set(a) | set(b))
    k = len(label_list)
    index = {label: i for i, label in enumerate(label_list)}
    n = len(a)
    observed = [[0.0] * k for _ in range(k)]
    for x, y in zip(a, b, strict=True):
        observed[index[x]][index[y]] += 1 / n
    row = [sum(observed[i]) for i in range(k)]
    col = [sum(observed[i][j] for i in range(k)) for j in range(k)]

    def weight(i: int, j: int) -> float:
        if weights is None:
            return 0.0 if i == j else 1.0
        gap = abs(i - j) / (k - 1) if k > 1 else 0.0
        return gap if weights == "linear" else gap**2

    observed_disagreement = sum(weight(i, j) * observed[i][j] for i in range(k) for j in range(k))
    expected_disagreement = sum(weight(i, j) * row[i] * col[j] for i in range(k) for j in range(k))
    if math.isclose(expected_disagreement, 0.0, abs_tol=1e-12):
        return math.nan
    return 1.0 - observed_disagreement / expected_disagreement


@dataclass(frozen=True)
class StabilityStats:
    n_groups: int
    mean_std: float
    flip_rate: float  # share of groups whose repeated values were not all identical


def stability(groups: Mapping[str, Sequence[float]]) -> StabilityStats:
    """Spread of repeated measurements. Groups with fewer than two values are ignored."""
    usable = [list(v) for v in groups.values() if len(v) >= 2]
    if not usable:
        return StabilityStats(0, math.nan, math.nan)
    stds = [statistics.stdev(v) for v in usable]
    flips = [len(set(v)) > 1 for v in usable]
    return StabilityStats(len(usable), sum(stds) / len(stds), sum(flips) / len(flips))


def bootstrap_mean_ci(
    values: Sequence[float],
    *,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Mean with a percentile bootstrap confidence interval: (mean, low, high)."""
    if not values:
        return (math.nan, math.nan, math.nan)
    mean = sum(values) / len(values)
    if len(values) == 1:
        return (mean, mean, mean)
    rng = random.Random(seed)
    n = len(values)
    means = sorted(sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot))
    low = means[int((alpha / 2) * n_boot)]
    high = means[min(n_boot - 1, int((1 - alpha / 2) * n_boot))]
    return (mean, low, high)
