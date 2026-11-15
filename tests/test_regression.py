from __future__ import annotations

import pytest

from llm_eval_harness import CaseRun, RunResult, compare_runs


def make_run(label: str, scores: dict[str, list[int]], *, rubric: str = "r", version: str = "1"):
    runs = [
        CaseRun(case_id=case_id, repeat=i, scores={"quality": s})
        for case_id, values in scores.items()
        for i, s in enumerate(values)
    ]
    return RunResult(
        label=label,
        rubric_name=rubric,
        rubric_version=version,
        scales={"quality": (1, 5)},
        judge="j",
        runs=runs,
    )


CASES = [f"c{i}" for i in range(30)]


def test_identical_runs_show_no_change() -> None:
    base = make_run("a", {c: [3] for c in CASES})
    comp = compare_runs(base, make_run("b", {c: [3] for c in CASES}))
    assert comp.rows[0].verdict == "no clear change"
    assert not comp.has_regression


def test_consistent_drop_is_a_regression() -> None:
    base = make_run("a", {c: [4] for c in CASES})
    cand = make_run("b", {c: [3] for c in CASES})
    comp = compare_runs(base, cand)
    row = comp.rows[0]
    assert row.verdict == "regression" and comp.has_regression
    assert row.mean_delta == pytest.approx(-1.0)
    assert row.worsened == 30 and row.improved == 0
    assert row.ci_high < 0


def test_consistent_gain_is_an_improvement() -> None:
    base = make_run("a", {c: [2] for c in CASES})
    cand = make_run("b", {c: [3] for c in CASES})
    assert compare_runs(base, cand).rows[0].verdict == "improvement"


def test_noisy_small_difference_is_not_called_a_regression() -> None:
    # Mean drops by 0.1 but the cases disagree about the direction.
    base = make_run("a", {c: [3] for c in CASES})
    cand = make_run("b", {c: [2 if i < 5 else 3] for i, c in enumerate(CASES[:25])})
    cand.runs += [CaseRun(case_id=c, repeat=0, scores={"quality": 4}) for c in CASES[25:]]
    row = compare_runs(base, cand, min_effect=0.05).rows[0]
    assert row.ci_low < 0 < row.ci_high
    assert row.verdict == "no clear change"


def test_repeats_are_averaged_per_case() -> None:
    base = make_run("a", {c: [2, 4] for c in CASES})
    cand = make_run("b", {c: [3, 3] for c in CASES})
    assert compare_runs(base, cand).rows[0].mean_delta == pytest.approx(0.0)


def test_errored_runs_are_ignored() -> None:
    base = make_run("a", {c: [3] for c in CASES})
    cand = make_run("b", {c: [3] for c in CASES})
    cand.runs[0] = CaseRun(case_id="c0", repeat=0, error="boom")
    comp = compare_runs(base, cand)
    assert comp.rows[0].n == 29
    assert any("only one run" in w for w in comp.warnings)


def test_different_rubrics_cannot_be_compared() -> None:
    with pytest.raises(ValueError, match="rubrics differ"):
        compare_runs(make_run("a", {"c": [3]}), make_run("b", {"c": [3]}, rubric="other"))


def test_version_and_judge_mismatch_only_warn() -> None:
    cand = make_run("b", {"c": [3]}, version="2")
    cand.judge = "other-judge"
    comp = compare_runs(make_run("a", {"c": [3]}), cand)
    assert len(comp.warnings) == 2
