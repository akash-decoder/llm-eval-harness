"""Command line interface: run, report, compare, invariance."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .clients import load_client
from .io import load_callable, load_cases
from .judge import LLMJudge
from .models import Rubric, RunResult
from .perturb import invariance_test, name_swaps
from .regression import compare_runs
from .report import render_comparison, render_invariance, render_run
from .runner import run_eval


def _emit(text: str, out: str | None) -> None:
    print(text)
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(text, encoding="utf-8")


def _cmd_run(args: argparse.Namespace) -> int:
    rubric = Rubric.from_file(args.rubric)
    cases = load_cases(args.data)
    judge = LLMJudge(load_client(args.client, args.model, args.temperature), rubric)
    target = load_callable(args.target) if args.target else None
    label = args.label or (args.target.rpartition(":")[2] if args.target else "recorded")
    result = run_eval(
        cases,
        judge,
        target=target,
        target_name=args.target,
        repeats=args.repeats,
        label=label,
        max_workers=args.workers,
    )
    out = args.out or f"runs/{label}.json"
    result.save(out)
    print(f"saved {out} ({len(result.runs) - len(result.error_runs)}/{len(result.runs)} ok)")
    return 1 if result.error_runs and not result.ok_runs else 0


def _cmd_report(args: argparse.Namespace) -> int:
    _emit(render_run(RunResult.load(args.run)), args.out)
    return 0


def _cmd_compare(args: argparse.Namespace) -> int:
    comparison = compare_runs(
        RunResult.load(args.baseline), RunResult.load(args.candidate), min_effect=args.min_effect
    )
    _emit(render_comparison(comparison), args.out)
    return 1 if args.fail_on_regression and comparison.has_regression else 0


def _cmd_invariance(args: argparse.Namespace) -> int:
    cases = load_cases(args.data)
    target = load_callable(args.target)
    names = [n.strip() for n in Path(args.names).read_text(encoding="utf-8").splitlines()]
    names = [n for n in names if n and not n.startswith("#")]
    perturbations = {"name swap": name_swaps(args.name_field, args.text_field, names)}
    report = invariance_test(
        cases,
        target,
        perturbations,
        tolerance=args.tolerance,
        max_excess_flip_rate=args.max_excess,
        max_workers=args.workers,
    )
    _emit(render_invariance(report), args.out)
    return 1 if args.fail_on_bias and not report.passed else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="llm-eval", description="Evaluate LLM judges and the systems they grade."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="judge recorded outputs, or run a target and judge it")
    run.add_argument("--rubric", required=True, help="rubric JSON file")
    run.add_argument("--data", required=True, help="cases JSONL file")
    run.add_argument("--client", required=True, help="anthropic, openai, or module:Factory")
    run.add_argument("--model", help="model name for anthropic/openai clients")
    run.add_argument(
        "--temperature",
        type=float,
        help="sampling temperature for the judge; unset by default (some models reject it)",
    )
    run.add_argument("--target", help="module:function (or file.py:function) to evaluate")
    run.add_argument("--repeats", type=int, default=1)
    run.add_argument("--label", help="name for this run (default: target function name)")
    run.add_argument("--out", help="where to save the run JSON (default: runs/<label>.json)")
    run.add_argument("--workers", type=int, default=4)
    run.set_defaults(func=_cmd_run)

    report = sub.add_parser("report", help="print a Markdown report for a saved run")
    report.add_argument("run")
    report.add_argument("--out")
    report.set_defaults(func=_cmd_report)

    compare = sub.add_parser("compare", help="paired comparison of two saved runs")
    compare.add_argument("baseline")
    compare.add_argument("candidate")
    compare.add_argument("--min-effect", type=float, default=0.1)
    compare.add_argument("--fail-on-regression", action="store_true")
    compare.add_argument("--out")
    compare.set_defaults(func=_cmd_compare)

    inv = sub.add_parser("invariance", help="swap names and check that outputs do not move")
    inv.add_argument("--data", required=True)
    inv.add_argument("--target", required=True)
    inv.add_argument("--name-field", required=True, help="input key holding the name")
    inv.add_argument(
        "--text-field",
        action="append",
        default=[],
        help="input key whose text mentions the name (repeatable)",
    )
    inv.add_argument("--names", required=True, help="text file, one replacement name per line")
    inv.add_argument("--tolerance", type=float, default=0.0)
    inv.add_argument("--max-excess", type=float, default=0.05)
    inv.add_argument("--workers", type=int, default=4)
    inv.add_argument("--fail-on-bias", action="store_true")
    inv.add_argument("--out")
    inv.set_defaults(func=_cmd_invariance)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except (ValueError, FileNotFoundError, KeyError, TypeError, AttributeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
