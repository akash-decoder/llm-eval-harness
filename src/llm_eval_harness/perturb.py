"""Counterfactual invariance tests.

Change something that should not matter (a candidate's name, the order of bullet points)
and check that the target's numeric outputs do not move. Because LLM targets are noisy
even on identical input, every case is also run twice unmodified. That control gives the
noise floor, and a perturbation only fails if it flips results more often than the floor.
"""

from __future__ import annotations

import copy
import re
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel

from .models import Case, jsonable

Perturbation = Callable[[dict[str, Any]], dict[str, Any]]
Extractor = Callable[[Any], Mapping[str, float]]


def numeric_fields(output: Any) -> dict[str, float]:
    """Default extractor: every top-level numeric value of a dict output."""
    if not isinstance(output, Mapping):
        raise TypeError("target output is not a dict; pass a custom `extract` function")
    return {
        str(k): float(v)
        for k, v in output.items()
        if isinstance(v, (int, float)) and not isinstance(v, bool)
    }


def replace_name(text: str, old: str, new: str) -> str:
    """Replace a full name, then any stand-alone first or last name, with the new name's parts."""
    text = text.replace(old, new)
    old_parts, new_parts = old.split(), new.split()
    if len(old_parts) >= 2 and len(new_parts) >= 2:
        for o, n in ((old_parts[0], new_parts[0]), (old_parts[-1], new_parts[-1])):
            text = re.sub(rf"\b{re.escape(o)}\b", n, text)
    return text


@dataclass(frozen=True)
class NameSwap:
    """Swap the candidate's name everywhere it appears in the input."""

    name_field: str
    text_fields: Sequence[str]
    replacement: str

    def __call__(self, case_input: dict[str, Any]) -> dict[str, Any]:
        old = case_input.get(self.name_field)
        if not isinstance(old, str) or not old:
            raise KeyError(f"input has no string field {self.name_field!r}")
        new_input = copy.deepcopy(case_input)
        new_input[self.name_field] = self.replacement
        for field in self.text_fields:
            value = new_input.get(field)
            if isinstance(value, str):
                new_input[field] = replace_name(value, old, self.replacement)
        return new_input


def name_swaps(
    name_field: str, text_fields: Sequence[str], replacements: Sequence[str]
) -> list[Perturbation]:
    return [NameSwap(name_field, tuple(text_fields), r) for r in replacements]


class InvarianceRow(BaseModel):
    family: str
    metric: str
    n: int
    mean_delta: float  # perturbed minus original
    mean_abs_delta: float
    max_abs_delta: float
    flip_rate: float
    control_flip_rate: float
    passed: bool


class InvarianceReport(BaseModel):
    rows: list[InvarianceRow]
    tolerance: float
    max_excess_flip_rate: float
    n_cases: int
    errors: list[str]

    @property
    def passed(self) -> bool:
        return all(r.passed for r in self.rows)


def invariance_test(
    cases: Sequence[Case],
    target: Callable[[dict[str, Any]], Any],
    perturbations: Mapping[str, Sequence[Perturbation]],
    *,
    extract: Extractor = numeric_fields,
    tolerance: float = 0.0,
    max_excess_flip_rate: float = 0.05,
    max_workers: int = 4,
) -> InvarianceReport:
    """Run each case unmodified (twice) and under every perturbation, then compare.

    A "flip" is a change larger than `tolerance` in one extracted metric. A family passes
    when its flip rate is no more than the control flip rate plus `max_excess_flip_rate`.
    """
    # Task = (case_id, kind, family, input). Perturbations that leave the input unchanged
    # are dropped (for example swapping in the candidate's own name).
    tasks: list[tuple[str, str, str, dict[str, Any]]] = []
    for case in cases:
        tasks.append((case.id, "base", "", case.input))
        tasks.append((case.id, "control", "", case.input))
        for family, perts in perturbations.items():
            for p in perts:
                changed = p(case.input)
                if changed != case.input:
                    tasks.append((case.id, "variant", family, changed))

    errors: list[str] = []

    def call(task: tuple[str, str, str, dict[str, Any]]) -> dict[str, float] | None:
        case_id, kind, family, inp = task
        try:
            return dict(extract(jsonable(target(inp))))
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{case_id} [{kind} {family}]: {type(exc).__name__}: {exc}".strip())
            return None

    if max_workers <= 1:
        results = [call(t) for t in tasks]
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            results = list(pool.map(call, tasks))

    base: dict[str, dict[str, float]] = {}
    control: dict[str, dict[str, float]] = {}
    variants: list[tuple[str, str, dict[str, float]]] = []
    for (case_id, kind, family, _), res in zip(tasks, results, strict=True):
        if res is None:
            continue
        if kind == "base":
            base[case_id] = res
        elif kind == "control":
            control[case_id] = res
        else:
            variants.append((case_id, family, res))

    def deltas(
        pairs: list[tuple[dict[str, float], dict[str, float]]],
    ) -> dict[str, list[float]]:
        out: dict[str, list[float]] = {}
        for original, changed in pairs:
            for metric in original.keys() & changed.keys():
                out.setdefault(metric, []).append(changed[metric] - original[metric])
        return out

    control_deltas = deltas([(base[c], control[c]) for c in base if c in control])
    control_flip = {
        m: sum(abs(d) > tolerance for d in ds) / len(ds) for m, ds in control_deltas.items()
    }

    rows: list[InvarianceRow] = []
    for family in perturbations:
        pairs = [(base[c], res) for c, fam, res in variants if fam == family and c in base]
        for metric, ds in sorted(deltas(pairs).items()):
            flip = sum(abs(d) > tolerance for d in ds) / len(ds)
            floor = control_flip.get(metric, 0.0)
            rows.append(
                InvarianceRow(
                    family=family,
                    metric=metric,
                    n=len(ds),
                    mean_delta=sum(ds) / len(ds),
                    mean_abs_delta=sum(abs(d) for d in ds) / len(ds),
                    max_abs_delta=max(abs(d) for d in ds),
                    flip_rate=flip,
                    control_flip_rate=floor,
                    passed=flip <= floor + max_excess_flip_rate,
                )
            )
    return InvarianceReport(
        rows=rows,
        tolerance=tolerance,
        max_excess_flip_rate=max_excess_flip_rate,
        n_cases=len(base),
        errors=sorted(errors),
    )
