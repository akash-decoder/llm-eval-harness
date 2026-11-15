"""Rubric-based LLM judge."""

from __future__ import annotations

import json
import re
from typing import Any, Protocol

from pydantic import BaseModel, Field

from .clients import LLMClient, client_name
from .models import Rubric


class JudgeError(Exception):
    """The judge could not produce a valid grade."""


class JudgeParseError(JudgeError):
    """The model replied, but not in the format the rubric asks for."""


class JudgeScore(BaseModel):
    scores: dict[str, int]
    rationales: dict[str, str] = Field(default_factory=dict)
    attempts: int = 1


class Judge(Protocol):
    name: str
    rubric: Rubric

    def grade(self, case_input: dict[str, Any], output: Any) -> JudgeScore: ...


SYSTEM_TEMPLATE = """\
You are a strict, consistent evaluator. Grade the OUTPUT against the rubric below, \
using the INPUT as context. Score each criterion independently, using only evidence \
visible in the INPUT and OUTPUT. Do not reward length, polish or a confident tone.
{instructions}
Criteria (integer scores):
{criteria_block}

Reply with one JSON object and nothing else, shaped like this:
{{"criteria": {{"<criterion id>": {{"rationale": "<one or two sentences>", "score": <integer>}}}}}}
Write each rationale before its score. Include every criterion id exactly once."""


def _stringify(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, indent=2, ensure_ascii=False)


def build_system_prompt(rubric: Rubric) -> str:
    blocks = []
    for c in rubric.criteria:
        lines = [f"- {c.id} ({c.scale_min}-{c.scale_max}): {c.description}"]
        for score in sorted(c.anchors):
            lines.append(f"    {score} = {c.anchors[score]}")
        blocks.append("\n".join(lines))
    instructions = f"\n{rubric.instructions.strip()}\n" if rubric.instructions.strip() else ""
    return SYSTEM_TEMPLATE.format(instructions=instructions, criteria_block="\n".join(blocks))


def build_user_prompt(case_input: dict[str, Any], output: Any) -> str:
    return (
        f"<input>\n{_stringify(case_input)}\n</input>\n\n<output>\n{_stringify(output)}\n</output>"
    )


_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


def _extract_json_object(text: str) -> dict[str, Any]:
    stripped = _FENCE.sub("", text.strip())
    decoder = json.JSONDecoder()
    start = stripped.find("{")
    while start != -1:
        try:
            obj, _ = decoder.raw_decode(stripped[start:])
        except json.JSONDecodeError:
            start = stripped.find("{", start + 1)
            continue
        if isinstance(obj, dict):
            return obj
        start = stripped.find("{", start + 1)
    raise JudgeParseError("no JSON object found in the reply")


def _as_int(value: Any, criterion_id: str) -> int:
    if isinstance(value, bool):
        raise JudgeParseError(f"{criterion_id}: score must be a number, got a boolean")
    if isinstance(value, (int, float)):
        number = float(value)
    elif isinstance(value, str):
        try:
            number = float(value.strip())
        except ValueError:
            raise JudgeParseError(f"{criterion_id}: score {value!r} is not a number") from None
    else:
        raise JudgeParseError(f"{criterion_id}: score must be a number")
    if number != int(number):
        raise JudgeParseError(f"{criterion_id}: score {value!r} is not a whole number")
    return int(number)


def parse_judge_response(text: str, rubric: Rubric) -> JudgeScore:
    data = _extract_json_object(text)
    criteria = data.get("criteria")
    if not isinstance(criteria, dict):
        raise JudgeParseError('missing the top-level "criteria" object')
    scores: dict[str, int] = {}
    rationales: dict[str, str] = {}
    for c in rubric.criteria:
        entry = criteria.get(c.id)
        if not isinstance(entry, dict) or "score" not in entry:
            raise JudgeParseError(f"missing score for criterion {c.id!r}")
        score = _as_int(entry["score"], c.id)
        if not c.scale_min <= score <= c.scale_max:
            raise JudgeParseError(f"{c.id}: score {score} outside {c.scale_min}-{c.scale_max}")
        scores[c.id] = score
        rationales[c.id] = str(entry.get("rationale", "")).strip()
    return JudgeScore(scores=scores, rationales=rationales)


class LLMJudge:
    """Grades an output against a rubric with an LLM, retrying when the reply is malformed."""

    def __init__(self, client: LLMClient, rubric: Rubric, *, max_retries: int = 2) -> None:
        self.client = client
        self.rubric = rubric
        self.max_retries = max_retries
        self.name = client_name(client)
        self._system = build_system_prompt(rubric)

    def grade(self, case_input: dict[str, Any], output: Any) -> JudgeScore:
        user = build_user_prompt(case_input, output)
        prompt = user
        last_error: JudgeParseError | None = None
        for attempt in range(1, self.max_retries + 2):
            reply = self.client.complete(self._system, prompt)
            try:
                result = parse_judge_response(reply, self.rubric)
            except JudgeParseError as exc:
                last_error = exc
                prompt = (
                    f"{user}\n\nYour previous reply was rejected: {exc}. "
                    "Reply again with only the JSON object described in the instructions."
                )
                continue
            result.attempts = attempt
            return result
        raise JudgeError(
            f"no valid grade after {self.max_retries + 1} attempts; last error: {last_error}"
        )
