.PHONY: install test lint demo clean

install:
	pip install -e ".[dev]"

test:
	pytest -q

lint:
	ruff check .
	ruff format --check .
	mypy

# Offline demo: synthetic data, heuristic judge (no API key, no network).
RUBRIC = examples/rubrics/screening.json
DATA = examples/data/gold.jsonl
JUDGE = examples/demo_judge.py:DemoJudgeClient

demo:
	llm-eval run --rubric $(RUBRIC) --data $(DATA) --client $(JUDGE) --repeats 3 --label calibration --out runs/calibration.json
	llm-eval report runs/calibration.json
	llm-eval run --rubric $(RUBRIC) --data $(DATA) --client $(JUDGE) --target examples/toy_target.py:screener_v1 --label v1 --out runs/v1.json
	llm-eval run --rubric $(RUBRIC) --data $(DATA) --client $(JUDGE) --target examples/toy_target.py:screener_v2 --label v2 --out runs/v2.json
	llm-eval compare runs/v1.json runs/v2.json
	llm-eval run --rubric $(RUBRIC) --data $(DATA) --client $(JUDGE) --target examples/toy_target.py:screener_v3_hallucinates --label v3 --out runs/v3.json
	-llm-eval compare runs/v2.json runs/v3.json --fail-on-regression
	llm-eval invariance --data $(DATA) --target examples/toy_target.py:screener_v2 \
		--name-field candidate_name --text-field resume_text --names examples/data/names.txt
	-llm-eval invariance --data $(DATA) --target examples/toy_target.py:screener_leaky \
		--name-field candidate_name --text-field resume_text --names examples/data/names.txt --fail-on-bias

clean:
	rm -rf runs .pytest_cache .mypy_cache .ruff_cache
