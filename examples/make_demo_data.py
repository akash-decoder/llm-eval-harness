"""Generate examples/data/gold.jsonl: synthetic resumes, recorded screening reports and
synthetic 'human' labels. Everything here is made up. Run: python examples/make_demo_data.py
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from demo_judge import heuristic_scores
from toy_target import screener_v1, screener_v2, screener_v3_hallucinates

REQUIREMENTS = ["Python", "FastAPI", "AWS", "LangChain", "RAG", "Docker"]

SKILL_LINES = {
    "Python": "Built Python data pipelines that process 1.2M records a day",
    "FastAPI": "Shipped FastAPI services behind an API gateway",
    "AWS": "Deployed workloads on AWS using ECS, S3 and SQS",
    "LangChain": "Prototyped document chains with LangChain and a vector store",
    "RAG": "Added RAG over internal wikis to cut support lookups",
    "Docker": "Containerised services with Docker and wired CI builds",
}
FILLER = [
    "Led a team of four through two product launches",
    "Wrote weekly status notes for non-technical stakeholders",
    "Mentored two junior colleagues",
    "Reduced page load time by caching hot queries",
    "Ran incident reviews after production outages",
]
FIRST = ["Maya", "Tomás", "Aisha", "Kenji", "Olga", "Daniel", "Priya", "Lucas", "Fatima", "Jonas"]
LAST = ["Okafor", "Lindqvist", "Rahman", "Tanaka", "Petrova", "Reyes", "Nair", "Weber", "Haddad"]
SCREENERS = [screener_v1, screener_v2, screener_v3_hallucinates]


def noisy(score: int, rng: random.Random) -> int:
    """Pretend human label: usually matches the heuristic, sometimes off by one."""
    if rng.random() < 0.3:
        score += rng.choice((-1, 1))
    return min(5, max(1, score))


def main() -> None:
    rng = random.Random(7)
    lines = []
    for i in range(40):
        name = f"{rng.choice(FIRST)} {rng.choice(LAST)}"
        have = rng.sample(REQUIREMENTS, k=rng.randint(3, len(REQUIREMENTS)))
        body = [SKILL_LINES[r] for r in REQUIREMENTS if r in have]
        body += rng.sample(FILLER, k=2)
        rng.shuffle(body)
        inp = {
            "candidate_name": name,
            "requirements": REQUIREMENTS,
            "resume_text": "\n".join([name, "", *body]),
        }
        output = rng.choice(SCREENERS)(inp)
        human = {k: noisy(v, rng) for k, v in heuristic_scores(inp, output).items()}
        lines.append(
            {
                "id": f"case-{i:03d}",
                "input": inp,
                "output": output,
                "human_scores": human,
                "tags": ["synthetic"],
            }
        )
    path = Path(__file__).parent / "data" / "gold.jsonl"
    path.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in lines) + "\n")
    print(f"wrote {len(lines)} cases to {path}")


if __name__ == "__main__":
    main()
