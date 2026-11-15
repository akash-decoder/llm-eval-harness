from __future__ import annotations

from typing import Any

import pytest

from conftest import make_case
from llm_eval_harness import Case, FunctionClient, LLMJudge, Rubric, RunResult, run_eval
from llm_eval_harness.io import load_cases


def cases(n: int = 3, **extra: Any) -> list[Case]:
    return [Case.model_validate(make_case(i, **extra)) for i in range(n)]


def test_judges_recorded_outputs_and_keeps_human_labels(constant_judge: LLMJudge) -> None:
    labelled = cases(2, human_scores={"accuracy": 4, "tone": 2})
    result = run_eval(labelled, constant_judge, label="recorded")
    assert [r.case_id for r in result.runs] == ["c0", "c1"]
    assert all(r.scores == {"accuracy": 4, "tone": 2} for r in result.runs)
    assert set(result.human_scores) == {"c0", "c1"}
    assert result.scales == {"accuracy": (1, 5), "tone": (1, 3)}


def test_target_output_is_generated_judged_and_human_labels_dropped(
    constant_judge: LLMJudge,
) -> None:
    result = run_eval(
        cases(2, human_scores={"accuracy": 1, "tone": 1}),
        constant_judge,
        target=lambda inp: {"echo": inp["q"]},
        target_name="echo",
    )
    assert result.runs[0].output == {"echo": "question 0"}
    assert result.human_scores == {}
    assert result.target == "echo"


def test_repeats_produce_one_run_per_case_and_repeat(constant_judge: LLMJudge) -> None:
    result = run_eval(cases(3), constant_judge, repeats=4, max_workers=1)
    assert len(result.runs) == 12
    assert [(r.case_id, r.repeat) for r in result.runs[:5]] == [
        ("c0", 0),
        ("c0", 1),
        ("c0", 2),
        ("c0", 3),
        ("c1", 0),
    ]


def test_one_failing_case_does_not_stop_the_run(constant_judge: LLMJudge) -> None:
    def flaky(inp: dict[str, Any]) -> str:
        if inp["q"].endswith("1"):
            raise RuntimeError("boom")
        return "fine"

    result = run_eval(cases(3), constant_judge, target=flaky)
    assert [r.error is None for r in result.runs] == [True, False, True]
    assert "RuntimeError: boom" in (result.runs[1].error or "")
    assert len(result.ok_runs) == 2 and len(result.error_runs) == 1


def test_case_without_output_or_target_is_recorded_as_error(constant_judge: LLMJudge) -> None:
    result = run_eval([Case(id="x", input={})], constant_judge)
    assert "no recorded output" in (result.runs[0].error or "")


def test_judge_errors_are_recorded(rubric: Rubric) -> None:
    judge = LLMJudge(FunctionClient(lambda s, u: "garbage"), rubric, max_retries=0)
    result = run_eval(cases(1), judge)
    assert "JudgeError" in (result.runs[0].error or "")


def test_threaded_run_matches_serial_run(constant_judge: LLMJudge) -> None:
    serial = run_eval(cases(8), constant_judge, repeats=2, max_workers=1)
    threaded = run_eval(cases(8), constant_judge, repeats=2, max_workers=4)
    assert [(r.case_id, r.repeat, r.scores) for r in serial.runs] == [
        (r.case_id, r.repeat, r.scores) for r in threaded.runs
    ]


def test_invalid_repeats(constant_judge: LLMJudge) -> None:
    with pytest.raises(ValueError):
        run_eval(cases(1), constant_judge, repeats=0)


def test_run_result_survives_a_save_load_round_trip(
    constant_judge: LLMJudge, tmp_path: Any
) -> None:
    result = run_eval(cases(2), constant_judge, label="rt")
    path = tmp_path / "nested" / "rt.json"
    result.save(path)
    assert RunResult.load(path) == result


def test_load_cases_rejects_duplicates_and_bad_lines(tmp_path: Any) -> None:
    dup = tmp_path / "dup.jsonl"
    dup.write_text('{"id": "a", "input": {}}\n{"id": "a", "input": {}}\n')
    with pytest.raises(ValueError, match="duplicate case id"):
        load_cases(dup)
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"id": "a"}\n')
    with pytest.raises(ValueError, match="bad.jsonl:1"):
        load_cases(bad)
    empty = tmp_path / "empty.jsonl"
    empty.write_text("\n")
    with pytest.raises(ValueError, match="no cases"):
        load_cases(empty)
