"""Shared repository paths and fixture-script loading."""

import runpy
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]


def load_script(path: str | Path) -> dict[str, Any]:
    """Load a repository generator/evaluator without sharing mutable globals.

    Constructors return fresh images/configs. Evaluator namespaces intentionally
    stay isolated because tests patch their policy and runtime functions.
    """
    return runpy.run_path(str(path))
