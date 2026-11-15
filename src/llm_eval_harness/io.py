"""Reading datasets and loading user-supplied callables."""

from __future__ import annotations

import importlib
import importlib.util
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .models import Case


def load_cases(path: str | Path) -> list[Case]:
    """Read a JSONL file with one Case per line. Blank lines are skipped."""
    cases: list[Case] = []
    seen: set[str] = set()
    with Path(path).open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                case = Case.model_validate_json(line)
            except ValueError as exc:
                raise ValueError(f"{path}:{lineno}: {exc}") from exc
            if case.id in seen:
                raise ValueError(f"{path}:{lineno}: duplicate case id {case.id!r}")
            seen.add(case.id)
            cases.append(case)
    if not cases:
        raise ValueError(f"{path}: no cases found")
    return cases


def load_callable(spec: str) -> Callable[..., Any]:
    """Resolve 'package.module:name' or 'path/to/file.py:name' to a callable."""
    if ":" not in spec:
        raise ValueError(f"expected 'module:name' or 'file.py:name', got {spec!r}")
    module_part, _, attr = spec.rpartition(":")
    if module_part.endswith(".py"):
        path = Path(module_part).resolve()
        if not path.exists():
            raise FileNotFoundError(path)
        mod_name = f"_llm_eval_user_{path.stem}"
        module = sys.modules.get(mod_name)
        if module is None:
            mod_spec = importlib.util.spec_from_file_location(mod_name, path)
            if mod_spec is None or mod_spec.loader is None:
                raise ImportError(f"cannot load {path}")
            module = importlib.util.module_from_spec(mod_spec)
            sys.modules[mod_name] = module
            mod_spec.loader.exec_module(module)
    else:
        module = importlib.import_module(module_part)
    try:
        obj = getattr(module, attr)
    except AttributeError as exc:
        raise AttributeError(f"{module_part} has no attribute {attr!r}") from exc
    if not callable(obj):
        raise TypeError(f"{spec} is not callable")
    return obj  # type: ignore[no-any-return]
