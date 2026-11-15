"""Data models shared across the harness."""

from __future__ import annotations

import dataclasses
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, model_validator


class Criterion(BaseModel):
    """One dimension a judge scores, on an integer scale."""

    id: str
    description: str
    scale_min: int = 1
    scale_max: int = 5
    anchors: dict[int, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check_scale(self) -> Criterion:
        if self.scale_min >= self.scale_max:
            raise ValueError(f"{self.id}: scale_min must be below scale_max")
        outside = [k for k in self.anchors if not self.scale_min <= k <= self.scale_max]
        if outside:
            raise ValueError(f"{self.id}: anchors {outside} fall outside the scale")
        return self

    @property
    def labels(self) -> list[int]:
        return list(range(self.scale_min, self.scale_max + 1))


class Rubric(BaseModel):
    name: str
    version: str = "1"
    instructions: str = ""
    criteria: list[Criterion]

    @model_validator(mode="after")
    def _check_criteria(self) -> Rubric:
        if not self.criteria:
            raise ValueError("a rubric needs at least one criterion")
        ids = [c.id for c in self.criteria]
        dupes = sorted({i for i in ids if ids.count(i) > 1})
        if dupes:
            raise ValueError(f"duplicate criterion ids: {dupes}")
        return self

    @property
    def ids(self) -> list[str]:
        return [c.id for c in self.criteria]

    def criterion(self, criterion_id: str) -> Criterion:
        for c in self.criteria:
            if c.id == criterion_id:
                return c
        raise KeyError(criterion_id)

    @classmethod
    def from_file(cls, path: str | Path) -> Rubric:
        return cls.model_validate_json(Path(path).read_text(encoding="utf-8"))


class Case(BaseModel):
    """One evaluation example.

    `output` holds a recorded output to judge. Leave it empty when a target
    function will generate the output. `human_scores` label the recorded
    output, one integer per criterion id.
    """

    id: str
    input: dict[str, Any]
    output: Any = None
    human_scores: dict[str, int] | None = None
    tags: list[str] = Field(default_factory=list)


class CaseRun(BaseModel):
    case_id: str
    repeat: int
    output: Any = None
    scores: dict[str, int] = Field(default_factory=dict)
    rationales: dict[str, str] = Field(default_factory=dict)
    error: str | None = None
    latency_s: float = 0.0


class RunResult(BaseModel):
    label: str
    rubric_name: str
    rubric_version: str
    scales: dict[str, tuple[int, int]]
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    target: str | None = None
    judge: str | None = None
    repeats: int = 1
    # Only filled when recorded outputs were judged; human labels describe those outputs.
    human_scores: dict[str, dict[str, int]] = Field(default_factory=dict)
    runs: list[CaseRun]

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.model_dump_json(indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> RunResult:
        return cls.model_validate_json(Path(path).read_text(encoding="utf-8"))

    @property
    def ok_runs(self) -> list[CaseRun]:
        return [r for r in self.runs if r.error is None]

    @property
    def error_runs(self) -> list[CaseRun]:
        return [r for r in self.runs if r.error is not None]


def jsonable(value: Any) -> Any:
    """Best-effort conversion of a target's return value into plain JSON types."""
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return dataclasses.asdict(value)
    # Round-trip so tuples, sets-as-lists etc. are stored the way they will be reloaded.
    return json.loads(json.dumps(value, default=str))
