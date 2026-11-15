"""llm-eval-harness: calibrate LLM judges, measure stability, test invariance, catch regressions."""

from .clients import AnthropicClient, FunctionClient, LLMClient, OpenAIClient, load_client
from .judge import Judge, JudgeError, JudgeScore, LLMJudge
from .metrics import bootstrap_mean_ci, cohens_kappa, stability
from .models import Case, CaseRun, Criterion, Rubric, RunResult
from .perturb import InvarianceReport, NameSwap, invariance_test, name_swaps
from .regression import Comparison, compare_runs
from .runner import run_eval

__version__ = "0.1.0"

__all__ = [
    "AnthropicClient",
    "Case",
    "CaseRun",
    "Comparison",
    "Criterion",
    "FunctionClient",
    "InvarianceReport",
    "Judge",
    "JudgeError",
    "JudgeScore",
    "LLMClient",
    "LLMJudge",
    "NameSwap",
    "OpenAIClient",
    "Rubric",
    "RunResult",
    "bootstrap_mean_ci",
    "cohens_kappa",
    "compare_runs",
    "invariance_test",
    "load_client",
    "name_swaps",
    "run_eval",
    "stability",
]
