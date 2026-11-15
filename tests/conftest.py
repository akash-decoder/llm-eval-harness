from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from llm_eval_harness import Criterion, FunctionClient, LLMJudge, Rubric

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"


@pytest.fixture
def rubric() -> Rubric:
    return Rubric(
        name="test-rubric",
        criteria=[
            Criterion(id="accuracy", description="Is it right?"),
            Criterion(id="tone", description="Is it polite?", scale_min=1, scale_max=3),
        ],
    )


def judge_reply(accuracy: int, tone: int) -> str:
    return json.dumps(
        {
            "criteria": {
                "accuracy": {"rationale": "r", "score": accuracy},
                "tone": {"rationale": "r", "score": tone},
            }
        }
    )


@pytest.fixture
def constant_judge(rubric: Rubric) -> LLMJudge:
    return LLMJudge(FunctionClient(lambda s, u: judge_reply(4, 2), name="constant"), rubric)


def make_case(i: int, **extra: Any) -> dict[str, Any]:
    return {"id": f"c{i}", "input": {"q": f"question {i}"}, "output": f"answer {i}", **extra}
