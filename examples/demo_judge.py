"""A deterministic stand-in for an LLM judge, used by the offline demo and the tests.

It is NOT a language model. It scores the toy screening reports with simple checks
(is the quoted evidence really in the resume? are all requirements covered? do the
questions mention a requirement?) and returns JSON in the format the real judge prompt
asks for. `jitter` flips a score by one point with that probability, reproducibly, so the
stability report has something to show.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import threading
from collections import defaultdict
from typing import Any

CRITERIA = ("evidence_grounding", "requirement_coverage", "question_specificity")


def _to_score(fraction: float) -> int:
    return 1 + round(4 * fraction)


def heuristic_scores(inp: dict[str, Any], out: dict[str, Any]) -> dict[str, int]:
    resume = inp["resume_text"]
    requirements = inp["requirements"]
    rows = out.get("requirements", [])

    quoted = [r["evidence"] for r in rows if r.get("evidence")]
    grounding = sum(q in resume for q in quoted) / len(quoted) if quoted else 0.0

    covered = {str(r["requirement"]).lower() for r in rows}
    coverage = sum(r.lower() in covered for r in requirements) / len(requirements)

    questions = out.get("questions", [])
    patterns = [re.compile(rf"\b{re.escape(r)}\b", re.IGNORECASE) for r in requirements]
    specific = sum(any(p.search(q) for p in patterns) for q in questions)
    specificity = specific / len(questions) if questions else 0.0

    return {
        "evidence_grounding": _to_score(grounding),
        "requirement_coverage": _to_score(coverage),
        "question_specificity": _to_score(specificity),
    }


_INPUT = re.compile(r"<input>\n(.*?)\n</input>", re.DOTALL)
_OUTPUT = re.compile(r"<output>\n(.*?)\n</output>", re.DOTALL)


class DemoJudgeClient:
    name = "heuristic-demo-judge"

    def __init__(self, jitter: float = 0.1, seed: int = 0) -> None:
        self.jitter = jitter
        self.seed = seed
        self._calls: dict[str, int] = defaultdict(int)
        self._lock = threading.Lock()

    def complete(self, system: str, user: str) -> str:
        inp_match, out_match = _INPUT.search(user), _OUTPUT.search(user)
        if not inp_match or not out_match:
            return "I could not find the input and output blocks."
        inp = json.loads(inp_match.group(1))
        out = json.loads(out_match.group(1))
        scores = heuristic_scores(inp, out)

        key = hashlib.sha256(user.encode()).hexdigest()
        with self._lock:
            count = self._calls[key]
            self._calls[key] += 1
        rng = random.Random(f"{self.seed}:{key}:{count}")
        criteria = {}
        for criterion in CRITERIA:
            score = scores[criterion]
            if rng.random() < self.jitter:
                score = min(5, max(1, score + rng.choice((-1, 1))))
            criteria[criterion] = {"rationale": f"heuristic check for {criterion}", "score": score}
        return json.dumps({"criteria": criteria})
