import json

import pytest

from conftest import judge_reply
from llm_eval_harness import FunctionClient, JudgeError, LLMJudge, Rubric
from llm_eval_harness.judge import (
    JudgeParseError,
    build_system_prompt,
    build_user_prompt,
    parse_judge_response,
)


def scripted(replies: list[str]) -> tuple[FunctionClient, list[str]]:
    seen: list[str] = []
    queue = list(replies)

    def fn(system: str, user: str) -> str:
        seen.append(user)
        return queue.pop(0)

    return FunctionClient(fn, name="scripted"), seen


def test_parses_plain_json(rubric: Rubric) -> None:
    result = parse_judge_response(judge_reply(4, 2), rubric)
    assert result.scores == {"accuracy": 4, "tone": 2}
    assert result.rationales["accuracy"] == "r"


def test_parses_fenced_json_with_chatter_around_it(rubric: Rubric) -> None:
    text = f"Sure, here you go:\n```json\n{judge_reply(5, 3)}\n```\nHope that helps."
    assert parse_judge_response(text, rubric).scores == {"accuracy": 5, "tone": 3}


def test_accepts_whole_number_floats_and_numeric_strings(rubric: Rubric) -> None:
    text = json.dumps(
        {"criteria": {"accuracy": {"score": 4.0, "rationale": ""}, "tone": {"score": "2"}}}
    )
    assert parse_judge_response(text, rubric).scores == {"accuracy": 4, "tone": 2}


@pytest.mark.parametrize(
    "payload",
    [
        "not json at all",
        json.dumps({"criteria": {"accuracy": {"score": 4}}}),  # tone missing
        json.dumps({"criteria": {"accuracy": {"score": 9}, "tone": {"score": 2}}}),  # out of range
        json.dumps({"criteria": {"accuracy": {"score": 3.5}, "tone": {"score": 2}}}),  # not whole
        json.dumps({"criteria": {"accuracy": {"score": True}, "tone": {"score": 2}}}),
        json.dumps({"scores": {"accuracy": 4, "tone": 2}}),  # wrong shape
    ],
)
def test_rejects_malformed_replies(payload: str, rubric: Rubric) -> None:
    with pytest.raises(JudgeParseError):
        parse_judge_response(payload, rubric)


def test_retries_after_a_bad_reply_and_tells_the_model_why(rubric: Rubric) -> None:
    client, seen = scripted(["oops", judge_reply(3, 1)])
    result = LLMJudge(client, rubric).grade({"q": 1}, "answer")
    assert result.scores == {"accuracy": 3, "tone": 1}
    assert result.attempts == 2
    assert "rejected" in seen[1] and "no JSON object" in seen[1]


def test_gives_up_after_max_retries(rubric: Rubric) -> None:
    client, seen = scripted(["bad", "worse", "still bad"])
    with pytest.raises(JudgeError, match="3 attempts"):
        LLMJudge(client, rubric, max_retries=2).grade({}, "x")
    assert len(seen) == 3


def test_prompts_contain_rubric_and_payload(rubric: Rubric) -> None:
    system = build_system_prompt(rubric)
    assert "accuracy (1-5)" in system and "tone (1-3)" in system
    user = build_user_prompt({"q": "hello"}, {"a": 1})
    assert '"q": "hello"' in user and user.index("<input>") < user.index("<output>")
