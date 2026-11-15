"""End-to-end tests of the CLI against the offline demo files."""

from __future__ import annotations

from pathlib import Path

import pytest

from conftest import EXAMPLES
from llm_eval_harness.cli import main
from llm_eval_harness.models import RunResult

RUBRIC = str(EXAMPLES / "rubrics" / "screening.json")
DATA = str(EXAMPLES / "data" / "gold.jsonl")
NAMES = str(EXAMPLES / "data" / "names.txt")
JUDGE = f"{EXAMPLES / 'demo_judge.py'}:DemoJudgeClient"


def target(name: str) -> str:
    return f"{EXAMPLES / 'toy_target.py'}:{name}"


def run_cli(*argv: str) -> int:
    return main(list(argv))


def test_calibration_run_and_report(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "cal.json"
    assert run_cli("run", "--rubric", RUBRIC, "--data", DATA, "--client", JUDGE,
                   "--repeats", "2", "--label", "cal", "--out", str(out)) == 0  # fmt: skip
    result = RunResult.load(out)
    assert len(result.runs) == 80 and not result.error_runs
    assert len(result.human_scores) == 40

    capsys.readouterr()
    assert run_cli("report", str(out), "--out", str(tmp_path / "report.md")) == 0
    text = capsys.readouterr().out
    assert "Judge vs human labels" in text and "Stability over 2 repeats" in text
    assert (tmp_path / "report.md").read_text().startswith("## Run `cal`")


def test_compare_flags_the_hallucinating_target(tmp_path: Path) -> None:
    runs = {}
    for name in ("screener_v2", "screener_v3_hallucinates"):
        path = tmp_path / f"{name}.json"
        assert run_cli("run", "--rubric", RUBRIC, "--data", DATA, "--client", JUDGE,
                       "--target", target(name), "--out", str(path)) == 0  # fmt: skip
        runs[name] = str(path)
    assert run_cli("compare", runs["screener_v2"], runs["screener_v3_hallucinates"]) == 0
    assert (
        run_cli(
            "compare", runs["screener_v2"], runs["screener_v3_hallucinates"], "--fail-on-regression"
        )
        == 1  # fmt: skip
    )
    # The other direction is an improvement, so the gate stays green.
    assert (
        run_cli(
            "compare", runs["screener_v3_hallucinates"], runs["screener_v2"], "--fail-on-regression"
        )
        == 0  # fmt: skip
    )


@pytest.mark.parametrize(("name", "expected"), [("screener_v2", 0), ("screener_leaky", 1)])
def test_invariance_gate(name: str, expected: int) -> None:
    code = run_cli("invariance", "--data", DATA, "--target", target(name),
                   "--name-field", "candidate_name", "--text-field", "resume_text",
                   "--names", NAMES, "--fail-on-bias")  # fmt: skip
    assert code == expected


def test_bad_input_returns_exit_code_2(capsys: pytest.CaptureFixture[str]) -> None:
    assert run_cli("run", "--rubric", "nope.json", "--data", DATA, "--client", JUDGE) == 2
    assert "error:" in capsys.readouterr().err
