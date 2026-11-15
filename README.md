# llm-eval-harness

**LLM judge calibration · score stability · counterfactual checks · regression testing**

[![ci](https://github.com/akash-decoder/llm-eval-harness/actions/workflows/ci.yml/badge.svg)](https://github.com/akash-decoder/llm-eval-harness/actions/workflows/ci.yml)

A small Python package and CLI for evaluating LLM judges and the systems they grade. It answers four questions that a single "average score" hides:

1. **Does the judge agree with human labels?** (calibration)
2. **Does it give the same score twice?** (stability)
3. **Does the system treat irrelevant changes as irrelevant?** (invariance, shown with a name-swap test)
4. **Did the last prompt change make things worse?** (regression, with a confidence interval)

It depends only on `pydantic`. Providers plug in through one method, `complete(system, user) -> str`. The CLI returns non-zero exit codes so checks 3 and 4 can gate a CI pipeline.

It was built to evaluate a resume-screening agent, but nothing in the harness is specific to recruiting. A case is a JSON input, an output, and optionally human scores.

## Quick start

```bash
git clone https://github.com/akash-decoder/llm-eval-harness.git
cd llm-eval-harness
```

```bash
pip install -e ".[dev]"
make demo        # offline: no API key, no network
pytest -q
```

The demo runs on **synthetic data with a heuristic stand-in for the judge** (`examples/demo_judge.py`, which is not a language model). The numbers below show what the reports look like and that the checks catch the problems they are built to catch. They say nothing about how good any real model is.

To use a real judge:

```bash
pip install -e ".[anthropic]"          # or ".[openai]"
export ANTHROPIC_API_KEY=...           # or OPENAI_API_KEY
llm-eval run --rubric my_rubric.json --data my_cases.jsonl \
  --client anthropic --model <model-name> --repeats 3
```

The hosted-provider adapters are covered by unit tests with fake clients and their keyword arguments were checked against the installed SDK signatures, but CI does not call the live APIs.

## The four checks

### 1. Judge vs human labels

Put human scores in your cases, run the judge over the recorded outputs, and read the report:

```bash
llm-eval run --rubric examples/rubrics/screening.json --data examples/data/gold.jsonl \
  --client examples/demo_judge.py:DemoJudgeClient --repeats 3 --label calibration
llm-eval report runs/calibration.json
```

| criterion | n | exact | within 1 | MAE | bias (judge-human) | QW kappa |
|---|---|---|---|---|---|---|
| evidence_grounding | 40 | 68% | 100% | 0.33 | +0.07 | 0.61 |
| requirement_coverage | 40 | 82% | 100% | 0.17 | +0.03 | 0.67 |
| question_specificity | 40 | 75% | 100% | 0.25 | +0.10 | 0.96 |

*Bias* is the mean of judge minus human, so a positive value means a lenient judge. *QW kappa* is quadratic-weighted Cohen's kappa: it discounts chance agreement and penalises a 1-vs-5 disagreement far more than a 3-vs-4 one. It is `n/a` when chance agreement is perfect (for example, both raters always give the same single score), because the statistic is undefined there. Accuracy figures on a gold set of 40 are rough; treat differences of a few points as noise.

### 2. Stability

`--repeats N` grades every case N times. With recorded outputs only the judge varies. With `--target`, the target is called again on every repeat, so the spread covers target and judge together.

Three repeats of the judge over recorded outputs:

| criterion | cases | mean std | cases with any change |
|---|---|---|---|
| evidence_grounding | 40 | 0.11 | 18% |
| requirement_coverage | 40 | 0.14 | 25% |
| question_specificity | 40 | 0.10 | 18% |

*Cases with any change* is the share of cases where the repeated scores were not all identical.

### 3. Invariance (name swap)

Replace the candidate's name (header, first name and surname wherever they appear in the listed text fields) with names from a file, run the target, and compare a numeric field of its output with the original run.

```bash
llm-eval invariance --data examples/data/gold.jsonl \
  --target examples/toy_target.py:screener_leaky \
  --name-field candidate_name --text-field resume_text \
  --names examples/data/names.txt --fail-on-bias
```

| perturbation | metric | n | mean delta | mean abs delta | max abs delta | flip rate | noise floor | result |
|---|---|---|---|---|---|---|---|---|
| name swap | overall_score | 480 | -0.004 | 0.221 | 0.500 | 44% | 0% | FAIL |

`screener_leaky` is deliberately broken: it adds 0.5 to the score when the name starts with A to M. Its *mean delta* is close to zero because the shifts go both ways, which is why the report also gives mean absolute delta and the flip rate. The well-behaved `screener_v2` gives `flip rate 0% against a 0% noise floor (pass)`.

LLM targets are not deterministic, so each case is also run twice on identical input. The flip rate between those two runs is the **noise floor**, and a perturbation only fails if it flips results more often than the floor plus `--max-excess` (default 5 points). Without that control, a temperature above zero would look like bias.

### 4. Regression between versions

Run two versions of the target through the same judge and compare them case by case:

```bash
llm-eval run ... --target examples/toy_target.py:screener_v2 --label v2
llm-eval run ... --target examples/toy_target.py:screener_v3_hallucinates --label v3
llm-eval compare runs/v2.json runs/v3.json --fail-on-regression
```

| criterion | paired n | v2 | v3 | delta | 95% CI | better / worse | verdict |
|---|---|---|---|---|---|---|---|
| evidence_grounding | 40 | 4.97 | 4.03 | -0.95 | [-1.20, -0.68] | 0 / 27 | REGRESSION |
| requirement_coverage | 40 | 4.90 | 4.97 | +0.07 | [-0.03, +0.17] | 4 / 1 | no clear change |
| question_specificity | 40 | 4.97 | 4.90 | -0.07 | [-0.17, +0.00] | 0 / 3 | no clear change |

The interval is a paired bootstrap over cases. A criterion is flagged as a regression only if the mean drop is at least `--min-effect` (default 0.1) and the whole interval is below zero. Otherwise it reports "no clear change". On small gold sets that is the common answer for small differences, and the tool does not pretend otherwise. Here `v3` invents evidence for requirements it cannot find in the resume, and the grounding criterion catches it.

## Data formats

**Rubric** (JSON): criteria with an id, description, integer scale and optional anchors per score. See `examples/rubrics/screening.json`.

**Cases** (JSONL, one per line):

```json
{"id": "case-001", "input": {"...": "..."}, "output": {"...": "..."}, "human_scores": {"accuracy": 4}, "tags": []}
```

`output` is optional when you pass `--target`; the target generates it. `human_scores` label the recorded `output`, so they are ignored when a target produces fresh outputs.

**Target**: any callable `dict -> output`, passed as `module:function` or `path/to/file.py:function`.

## Library use

```python
from llm_eval_harness import Case, LLMJudge, Rubric, AnthropicClient, run_eval, compare_runs

rubric = Rubric.from_file("my_rubric.json")
judge = LLMJudge(AnthropicClient("<model-name>"), rubric)
cases = [Case(id="a", input={"q": "..."})]

baseline = run_eval(cases, judge, target=agent_v1, label="v1", repeats=3)
candidate = run_eval(cases, judge, target=agent_v2, label="v2", repeats=3)
print(compare_runs(baseline, candidate).has_regression)
```

To add a provider, write a class with `complete(system: str, user: str) -> str` and an optional `name` attribute.

## Design notes and limits

- **The judge prompt asks for a rationale before each score** and the reply is parsed strictly: JSON only, every criterion present, integers inside the scale. A malformed reply is retried with the parse error appended; after `max_retries` the case is recorded as an error and the run continues. Failures are counted in the report, not hidden.
- **A judge is a measuring instrument, not ground truth.** Calibrate it against human labels before trusting its scores, and re-check when the judge model or the rubric changes. `compare` warns when two runs used different judges or rubric versions. When you can, grade with a different model family than the one being graded.
- **The name-swap test covers names only.** It does not touch email addresses, pronouns, photos, affiliations or other proxies. It shows that a target is sensitive to names. A pass does not show the target is fair.
- **Statistics are deliberately simple:** percentile bootstrap, Cohen's kappa, no numpy. They are checked against hand-computed values in `tests/test_metrics.py`.
- **Not built yet:** pairwise (A vs B) judging with position swapping, response caching, cost and latency tracking per run, async clients.

## Layout

```
src/llm_eval_harness/
  models.py      rubric, case and run schemas
  clients.py     provider adapters and a function wrapper
  judge.py       prompt building, strict parsing, retries
  runner.py      run target + judge over cases, with repeats and threads
  metrics.py     kappa, agreement, stability, bootstrap
  perturb.py     name swap and the invariance test with a noise floor
  regression.py  paired comparison of two runs
  report.py      Markdown rendering
  cli.py         llm-eval run | report | compare | invariance
examples/        rubric, synthetic data, toy targets, offline demo judge
tests/           unit and CLI tests, no network
```

## Development

```bash
make test    # pytest
make lint    # ruff check, ruff format --check, mypy --strict
```

Python 3.10 or newer. MIT licensed.
