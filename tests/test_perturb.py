from __future__ import annotations

import itertools
from typing import Any

import pytest

from llm_eval_harness import Case, NameSwap, invariance_test, name_swaps
from llm_eval_harness.perturb import numeric_fields, replace_name

NAMES = ["Amara Osei", "Rohan Mehta", "Elena Marković", "Henrik Solberg"]


def make_cases(n: int = 10) -> list[Case]:
    firsts = ["Ada", "Zoe", "Ben", "Mia", "Leo", "Ivy", "Max", "Eva", "Kai", "Noa"]
    cases = []
    for i in range(n):
        name = f"{firsts[i % 10]} Walker"
        text = f"{name}\n{firsts[i % 10]} led a team.\nPython, AWS\nWalker Labs intern"
        cases.append(
            Case(id=f"c{i}", input={"candidate_name": name, "resume_text": text, "years": i})
        )
    return cases


def fair_target(inp: dict[str, Any]) -> dict[str, Any]:
    return {"score": float(inp["years"]), "label": "ok"}


def leaky_target(inp: dict[str, Any]) -> dict[str, Any]:
    bonus = 1.0 if inp["candidate_name"].startswith(("A", "R")) else 0.0
    return {"score": float(inp["years"]) + bonus}


def perturbations() -> dict[str, list[Any]]:
    return {"name swap": name_swaps("candidate_name", ["resume_text"], NAMES)}


def test_replace_name_handles_full_first_and_last_name() -> None:
    text = "Ada Walker\nAda built it.\nWalker Labs\nawalker@example.com"
    out = replace_name(text, "Ada Walker", "Rohan Mehta")
    assert out == "Rohan Mehta\nRohan built it.\nMehta Labs\nawalker@example.com"


def test_name_swap_changes_only_the_named_fields() -> None:
    inp = {"candidate_name": "Ada Walker", "resume_text": "Ada Walker, Python", "years": 3}
    swapped = NameSwap("candidate_name", ["resume_text"], "Rohan Mehta")(inp)
    assert swapped == {
        "candidate_name": "Rohan Mehta",
        "resume_text": "Rohan Mehta, Python",
        "years": 3,
    }
    assert inp["candidate_name"] == "Ada Walker"  # original untouched


def test_name_swap_requires_the_name_field() -> None:
    with pytest.raises(KeyError):
        NameSwap("candidate_name", [], "X Y")({"resume_text": "text"})


def test_numeric_fields_skips_non_numbers_and_booleans() -> None:
    assert numeric_fields({"a": 1, "b": 2.5, "c": "x", "d": True, "e": [1]}) == {"a": 1.0, "b": 2.5}
    with pytest.raises(TypeError):
        numeric_fields("not a dict")


def test_fair_target_passes() -> None:
    report = invariance_test(make_cases(), fair_target, perturbations(), max_workers=1)
    assert report.passed
    (row,) = report.rows
    assert row.metric == "score" and row.flip_rate == 0.0 and row.n == 40


def test_leaky_target_is_caught() -> None:
    report = invariance_test(make_cases(), leaky_target, perturbations(), max_workers=1)
    assert not report.passed
    (row,) = report.rows
    assert row.flip_rate > 0.3
    assert row.max_abs_delta == 1.0


def test_noise_floor_stops_a_noisy_but_unbiased_target_from_failing() -> None:
    counter = itertools.count()

    def noisy(inp: dict[str, Any]) -> dict[str, float]:
        # Changes on every call regardless of the input, like a temperature > 0 model.
        return {"score": float(next(counter) % 2)}

    report = invariance_test(make_cases(), noisy, perturbations(), max_workers=1)
    (row,) = report.rows
    assert row.control_flip_rate > 0.0
    assert row.flip_rate > 0.0


def test_swapping_in_the_candidates_own_name_is_skipped() -> None:
    cases = [Case(id="a", input={"candidate_name": "Rohan Mehta", "resume_text": "Rohan Mehta"})]
    perts = {"swap": name_swaps("candidate_name", ["resume_text"], ["Rohan Mehta"])}
    report = invariance_test(cases, lambda inp: {"s": 1.0}, perts, max_workers=1)
    assert report.rows == []


def test_target_failures_are_reported_not_raised() -> None:
    def broken(inp: dict[str, Any]) -> dict[str, float]:
        raise RuntimeError("down")

    report = invariance_test(make_cases(2), broken, perturbations(), max_workers=1)
    assert report.rows == [] and report.errors
    assert "RuntimeError: down" in report.errors[0]


def test_tolerance_absorbs_small_changes() -> None:
    def slightly_off(inp: dict[str, Any]) -> dict[str, float]:
        return {"score": float(inp["years"]) + (0.01 if "Amara" in inp["candidate_name"] else 0)}

    strict = invariance_test(make_cases(), slightly_off, perturbations(), max_workers=1)
    lenient = invariance_test(
        make_cases(), slightly_off, perturbations(), tolerance=0.05, max_workers=1
    )
    assert not strict.passed and lenient.passed
