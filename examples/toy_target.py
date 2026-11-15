"""Toy 'screening agents' used by the offline demo. No LLM involved.

They stand in for a real agent so the harness can be exercised without API keys:

* screener_v1: naive, checks only the first 4 requirements and asks generic questions
* screener_v2: checks everything and asks targeted questions
* screener_v3_hallucinates: like v2, but invents evidence it cannot find in the resume
* screener_leaky: like v2, but the candidate's name nudges the score (deliberately broken)
"""

from __future__ import annotations

import re
from typing import Any

GENERIC_QUESTIONS = ["Tell me about yourself.", "What are your strengths?"]


def find_evidence(requirement: str, resume_text: str) -> str:
    """First resume line that mentions the requirement as a whole word, else ''."""
    pattern = re.compile(rf"\b{re.escape(requirement)}\b", re.IGNORECASE)
    for line in resume_text.splitlines():
        if pattern.search(line):
            return line.strip()
    return ""


def _targeted_questions(missing: list[str], matched: list[str]) -> list[str]:
    questions = [
        f"Your resume doesn't mention {req}. How have you worked with it, or something close?"
        for req in missing
    ]
    for req in matched:
        if len(questions) >= 3:
            break
        questions.append(f"Describe the hardest problem you solved with {req}.")
    return questions


def _screen(
    inp: dict[str, Any], *, limit: int | None = None, targeted: bool = True, invent: bool = False
) -> dict[str, Any]:
    requirements = list(inp["requirements"])[:limit]
    rows = []
    for req in requirements:
        evidence = find_evidence(req, inp["resume_text"])
        if evidence:
            rows.append({"requirement": req, "score": 4, "evidence": evidence})
        elif invent:
            rows.append(
                {
                    "requirement": req,
                    "score": 3,
                    "evidence": f"Hands-on {req} experience across several projects",
                }
            )
        else:
            rows.append({"requirement": req, "score": 1, "evidence": ""})
    matched = [r["requirement"] for r in rows if r["evidence"] and r["score"] == 4]
    missing = [r["requirement"] for r in rows if not r["evidence"]]
    questions = _targeted_questions(missing, matched) if targeted else list(GENERIC_QUESTIONS)
    overall = round(sum(r["score"] for r in rows) / len(rows), 2) if rows else 0.0
    return {"overall_score": overall, "requirements": rows, "questions": questions}


def screener_v1(inp: dict[str, Any]) -> dict[str, Any]:
    return _screen(inp, limit=4, targeted=False)


def screener_v2(inp: dict[str, Any]) -> dict[str, Any]:
    return _screen(inp)


def screener_v3_hallucinates(inp: dict[str, Any]) -> dict[str, Any]:
    return _screen(inp, invent=True)


def screener_leaky(inp: dict[str, Any]) -> dict[str, Any]:
    out = _screen(inp)
    if inp["candidate_name"][:1].upper() in "ABCDEFGHIJKLM":
        out["overall_score"] = round(out["overall_score"] + 0.5, 2)
    return out
