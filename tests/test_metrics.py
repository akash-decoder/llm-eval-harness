import math

import pytest

from llm_eval_harness.metrics import (
    bootstrap_mean_ci,
    cohens_kappa,
    exact_agreement,
    mean_abs_error,
    mean_signed_difference,
    stability,
    within_agreement,
)


def test_simple_agreement_numbers() -> None:
    a, b = [1, 2, 3, 4], [1, 2, 4, 2]
    assert exact_agreement(a, b) == 0.5
    assert within_agreement(a, b) == 0.75
    assert mean_abs_error(a, b) == pytest.approx(0.75)
    assert mean_signed_difference(a, b) == pytest.approx(0.25)


def test_length_mismatch_and_empty_input_are_rejected() -> None:
    with pytest.raises(ValueError):
        exact_agreement([1, 2], [1])
    with pytest.raises(ValueError):
        exact_agreement([], [])


def test_kappa_matches_textbook_example() -> None:
    # 20 yes/yes, 5 yes/no, 10 no/yes, 15 no/no: observed 0.7, chance 0.5, kappa 0.4.
    a = [1] * 20 + [1] * 5 + [0] * 10 + [0] * 15
    b = [1] * 20 + [0] * 5 + [1] * 10 + [0] * 15
    assert cohens_kappa(a, b) == pytest.approx(0.4)


def test_quadratic_weighted_kappa_hand_computed() -> None:
    # Worked by hand for labels 1..3: observed disagreement 1/12, expected 1/4, kappa 2/3.
    assert cohens_kappa([1, 2, 3], [1, 2, 2], labels=[1, 2, 3], weights="quadratic") == (
        pytest.approx(2 / 3)
    )


def test_kappa_is_one_for_identical_ratings() -> None:
    ratings = [1, 2, 3, 4, 5, 3, 2]
    for weights in (None, "linear", "quadratic"):
        assert cohens_kappa(ratings, ratings, weights=weights) == pytest.approx(1.0)


def test_weighted_kappa_forgives_near_misses_more() -> None:
    a = [1, 2, 3, 4, 5, 1, 2, 3, 4, 5]
    near = [2, 3, 4, 5, 4, 2, 1, 2, 5, 4]  # every rating off by one
    plain = cohens_kappa(a, near)
    weighted = cohens_kappa(a, near, labels=range(1, 6), weights="quadratic")
    assert weighted > plain


def test_kappa_is_nan_when_chance_agreement_is_perfect() -> None:
    assert math.isnan(cohens_kappa([3, 3, 3], [3, 3, 3], labels=[1, 2, 3, 4, 5]))


def test_stability_counts_changed_groups() -> None:
    s = stability({"a": [3, 3, 3], "b": [2, 4, 3], "c": [5]})
    assert s.n_groups == 2  # the single-value group is ignored
    assert s.flip_rate == 0.5
    assert s.mean_std == pytest.approx(0.5)  # (0 + 1) / 2


def test_stability_without_repeats_is_nan() -> None:
    assert math.isnan(stability({"a": [1], "b": [2]}).mean_std)


def test_bootstrap_ci_brackets_the_mean_and_is_reproducible() -> None:
    values = [0.0, 1.0, 1.0, 2.0, 1.0, 0.0, 2.0, 1.0]
    mean, low, high = bootstrap_mean_ci(values, seed=3)
    assert low <= mean <= high
    assert (mean, low, high) == bootstrap_mean_ci(values, seed=3)


def test_bootstrap_ci_for_constant_values_is_a_point() -> None:
    assert bootstrap_mean_ci([2.0] * 10) == (2.0, 2.0, 2.0)
