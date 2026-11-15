"""Run a judge (and optionally a target system) over a set of cases."""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from .judge import Judge
from .models import Case, CaseRun, RunResult, jsonable

Target = Callable[[dict[str, Any]], Any]


def run_eval(
    cases: Sequence[Case],
    judge: Judge,
    *,
    target: Target | None = None,
    target_name: str | None = None,
    repeats: int = 1,
    label: str = "run",
    max_workers: int = 4,
) -> RunResult:
    """Grade every case `repeats` times.

    With a `target`, each repeat calls the target again and judges the fresh output, so
    stability covers target and judge together. Without one, the recorded `case.output`
    is judged and only judge noise varies between repeats. A failure in one case is
    recorded on that CaseRun and does not stop the run.
    """
    if repeats < 1:
        raise ValueError("repeats must be at least 1")

    def one(task: tuple[Case, int]) -> CaseRun:
        case, repeat = task
        started = time.perf_counter()
        try:
            output = target(case.input) if target is not None else case.output
            if output is None:
                raise ValueError("case has no recorded output and no target was given")
            output = jsonable(output)
            graded = judge.grade(case.input, output)
            return CaseRun(
                case_id=case.id,
                repeat=repeat,
                output=output,
                scores=graded.scores,
                rationales=graded.rationales,
                latency_s=time.perf_counter() - started,
            )
        except Exception as exc:  # noqa: BLE001 - one bad case must not sink the run
            return CaseRun(
                case_id=case.id,
                repeat=repeat,
                error=f"{type(exc).__name__}: {exc}",
                latency_s=time.perf_counter() - started,
            )

    tasks = [(case, repeat) for case in cases for repeat in range(repeats)]
    if max_workers <= 1:
        runs = [one(t) for t in tasks]
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            runs = list(pool.map(one, tasks))

    human = (
        {c.id: c.human_scores for c in cases if c.human_scores is not None}
        if target is None
        else {}
    )
    return RunResult(
        label=label,
        rubric_name=judge.rubric.name,
        rubric_version=judge.rubric.version,
        scales={c.id: (c.scale_min, c.scale_max) for c in judge.rubric.criteria},
        target=target_name,
        judge=judge.name,
        repeats=repeats,
        human_scores=human,
        runs=runs,
    )
