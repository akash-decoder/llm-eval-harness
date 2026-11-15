"""Compare two runs of the same rubric and flag statistically clear regressions."""

from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel

from .metrics import bootstrap_mean_ci
from .models import RunResult

Verdict = Literal["regression", "improvement", "no clear change"]


class CriterionComparison(BaseModel):
    criterion: str
    n: int
    baseline_mean: float
    candidate_mean: float
    mean_delta: float
    ci_low: float
    ci_high: float
    improved: int
    worsened: int
    verdict: Verdict


class Comparison(BaseModel):
    baseline: str
    candidate: str
    rows: list[CriterionComparison]
    unpaired_cases: int
    warnings: list[str]

    @property
    def has_regression(self) -> bool:
        return any(r.verdict == "regression" for r in self.rows)


def per_case_means(run: RunResult, criterion: str) -> dict[str, float]:
    """Mean score per case across repeats, skipping runs that errored."""
    buckets: dict[str, list[int]] = {}
    for r in run.ok_runs:
        if criterion in r.scores:
            buckets.setdefault(r.case_id, []).append(r.scores[criterion])
    return {case_id: sum(v) / len(v) for case_id, v in buckets.items()}


def compare_runs(
    baseline: RunResult,
    candidate: RunResult,
    *,
    min_effect: float = 0.1,
    n_boot: int = 2000,
    seed: int = 0,
) -> Comparison:
    """Paired comparison per criterion.

    A criterion is a regression when the candidate is worse by at least `min_effect`
    points on average and the 95% bootstrap interval of the paired difference sits below
    zero. Improvements mirror that. Anything else is "no clear change", which on a small
    gold set is the usual outcome for small differences.
    """
    if baseline.rubric_name != candidate.rubric_name:
        raise ValueError(
            f"rubrics differ: {baseline.rubric_name!r} vs {candidate.rubric_name!r}; "
            "scores are not comparable"
        )
    warnings: list[str] = []
    if baseline.rubric_version != candidate.rubric_version:
        warnings.append(
            f"rubric versions differ ({baseline.rubric_version} vs {candidate.rubric_version})"
        )
    if baseline.judge != candidate.judge:
        warnings.append(f"judges differ ({baseline.judge} vs {candidate.judge})")

    rows: list[CriterionComparison] = []
    unpaired = 0
    for criterion in baseline.scales:
        if criterion not in candidate.scales:
            warnings.append(f"criterion {criterion!r} missing from candidate run")
            continue
        base = per_case_means(baseline, criterion)
        cand = per_case_means(candidate, criterion)
        shared = sorted(base.keys() & cand.keys())
        unpaired = max(unpaired, len(base.keys() ^ cand.keys()))
        if not shared:
            continue
        diffs = [cand[c] - base[c] for c in shared]
        mean, low, high = bootstrap_mean_ci(diffs, n_boot=n_boot, seed=seed)
        if high < 0 and mean <= -min_effect:
            verdict: Verdict = "regression"
        elif low > 0 and mean >= min_effect:
            verdict = "improvement"
        else:
            verdict = "no clear change"
        rows.append(
            CriterionComparison(
                criterion=criterion,
                n=len(shared),
                baseline_mean=sum(base[c] for c in shared) / len(shared),
                candidate_mean=sum(cand[c] for c in shared) / len(shared),
                mean_delta=mean,
                ci_low=low,
                ci_high=high,
                improved=sum(d > 0 for d in diffs),
                worsened=sum(d < 0 for d in diffs),
                verdict=verdict,
            )
        )
    if unpaired:
        warnings.append(f"{unpaired} case(s) appear in only one run and were left out")
    if any(math.isnan(r.mean_delta) for r in rows):
        warnings.append("some comparisons produced NaN")
    return Comparison(
        baseline=baseline.label,
        candidate=candidate.label,
        rows=rows,
        unpaired_cases=unpaired,
        warnings=warnings,
    )
