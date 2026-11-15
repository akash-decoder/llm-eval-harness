"""Markdown rendering for runs, comparisons and invariance tests."""

from __future__ import annotations

import math

from .metrics import (
    cohens_kappa,
    exact_agreement,
    mean_abs_error,
    mean_signed_difference,
    stability,
    within_agreement,
)
from .models import RunResult
from .perturb import InvarianceReport
from .regression import Comparison


def _fmt(x: float, digits: int = 2) -> str:
    return "n/a" if math.isnan(x) else f"{x:.{digits}f}"


def _pct(x: float) -> str:
    return "n/a" if math.isnan(x) else f"{100 * x:.0f}%"


def _table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def agreement_section(run: RunResult) -> str | None:
    """Judge vs human labels. Uses the first successful repeat of each case."""
    if not run.human_scores:
        return None
    first: dict[str, dict[str, int]] = {}
    for r in run.ok_runs:
        first.setdefault(r.case_id, r.scores)
    rows: list[list[str]] = []
    pooled_judge: list[int] = []
    pooled_human: list[int] = []
    for criterion, (lo, hi) in run.scales.items():
        judge, human = [], []
        for case_id, labels in run.human_scores.items():
            if case_id in first and criterion in labels and criterion in first[case_id]:
                judge.append(first[case_id][criterion])
                human.append(labels[criterion])
        if not judge:
            continue
        pooled_judge += judge
        pooled_human += human
        labels_range = range(lo, hi + 1)
        rows.append(
            [
                criterion,
                str(len(judge)),
                _pct(exact_agreement(judge, human)),
                _pct(within_agreement(judge, human)),
                _fmt(mean_abs_error(judge, human)),
                f"{mean_signed_difference(judge, human):+.2f}",
                _fmt(cohens_kappa(judge, human, labels=labels_range, weights="quadratic")),
            ]
        )
    if not rows:
        return None
    header = ["criterion", "n", "exact", "within 1", "MAE", "bias (judge-human)", "QW kappa"]
    note = (
        "Bias above zero means the judge is more lenient than the human labels. "
        "QW kappa is quadratic-weighted Cohen's kappa; n/a means chance agreement is "
        "perfect, so the statistic is undefined."
    )
    return "### Judge vs human labels\n\n" + _table(header, rows) + "\n\n" + note


def stability_section(run: RunResult) -> str | None:
    if run.repeats < 2:
        return None
    rows: list[list[str]] = []
    for criterion in run.scales:
        groups: dict[str, list[float]] = {}
        for r in run.ok_runs:
            if criterion in r.scores:
                groups.setdefault(r.case_id, []).append(r.scores[criterion])
        s = stability(groups)
        rows.append([criterion, str(s.n_groups), _fmt(s.mean_std), _pct(s.flip_rate)])
    what = "target and judge" if run.target else "judge only, recorded outputs"
    return (
        f"### Stability over {run.repeats} repeats: {what}\n\n"
        + _table(["criterion", "cases", "mean std", "cases with any change"], rows)
        + "\n\nLower is better. `mean std` is the average per-case sample standard deviation "
        "across repeats."
    )


def render_run(run: RunResult) -> str:
    total = len(run.runs)
    failed = len(run.error_runs)
    means = []
    for criterion in run.scales:
        vals = [r.scores[criterion] for r in run.ok_runs if criterion in r.scores]
        means.append([criterion, _fmt(sum(vals) / len(vals)) if vals else "n/a"])
    parts = [
        f"## Run `{run.label}`",
        f"Rubric `{run.rubric_name}` v{run.rubric_version}, judge `{run.judge}`"
        + (f", target `{run.target}`" if run.target else ", recorded outputs")
        + f". {total - failed}/{total} case-runs succeeded.",
        "### Mean scores\n\n" + _table(["criterion", "mean"], means),
    ]
    for section in (agreement_section(run), stability_section(run)):
        if section:
            parts.append(section)
    if failed:
        sample = "\n".join(f"- `{r.case_id}`: {r.error}" for r in run.error_runs[:5])
        parts.append(f"### Failures ({failed})\n\n{sample}")
    return "\n\n".join(parts) + "\n"


def render_comparison(comp: Comparison) -> str:
    rows = [
        [
            r.criterion,
            str(r.n),
            _fmt(r.baseline_mean),
            _fmt(r.candidate_mean),
            f"{r.mean_delta:+.2f}",
            f"[{r.ci_low:+.2f}, {r.ci_high:+.2f}]",
            f"{r.improved} / {r.worsened}",
            r.verdict.upper() if r.verdict == "regression" else r.verdict,
        ]
        for r in comp.rows
    ]
    header = [
        "criterion",
        "paired n",
        comp.baseline,
        comp.candidate,
        "delta",
        "95% CI",
        "better / worse",
        "verdict",
    ]
    out = f"## `{comp.candidate}` vs `{comp.baseline}`\n\n" + _table(header, rows)
    out += "\n\nCI is a paired bootstrap over cases. Small gold sets give wide intervals."
    if comp.warnings:
        out += "\n\n" + "\n".join(f"- warning: {w}" for w in comp.warnings)
    return out + "\n"


def render_invariance(rep: InvarianceReport) -> str:
    rows = [
        [
            r.family,
            r.metric,
            str(r.n),
            f"{r.mean_delta:+.3f}",
            _fmt(r.mean_abs_delta, 3),
            _fmt(r.max_abs_delta, 3),
            _pct(r.flip_rate),
            _pct(r.control_flip_rate),
            "pass" if r.passed else "FAIL",
        ]
        for r in rep.rows
    ]
    header = [
        "perturbation",
        "metric",
        "n",
        "mean delta",
        "mean abs delta",
        "max abs delta",
        "flip rate",
        "noise floor",
        "result",
    ]
    out = (
        f"## Invariance test ({rep.n_cases} cases)\n\n"
        + _table(header, rows)
        + f"\n\nA flip is a change larger than {rep.tolerance:g}. The noise floor is the flip "
        "rate between two runs of the same unmodified input. A perturbation fails when its "
        f"flip rate exceeds the floor by more than {rep.max_excess_flip_rate:.0%}."
    )
    if rep.errors:
        out += f"\n\n{len(rep.errors)} target call(s) failed, for example: {rep.errors[0]}"
    return out + "\n"
