"""
Load selected functions from a frozen pipeline script as an importable
reference module, for differential testing against the clean-room
package.

The frozen scripts are single-file, several thousand lines long, and
execute work at import time through ``main``. Importing one directly is
therefore not an option. This helper extracts only the names requested
plus their transitive module-level dependencies, and executes that
subset as a synthetic module.

The frozen scripts are not part of this repository. Point
``ADNI_FROZEN_SCRIPT_DIR`` at the directory holding them to enable the
differential tests; without it they are skipped.

No ADNI data is read here. Only the source text of the scripts.
"""

from __future__ import annotations

import ast
import importlib.util
import os
import sys
import types
from pathlib import Path
from typing import Final


FROZEN_DIR_ENV_VAR: Final[str] = "ADNI_FROZEN_SCRIPT_DIR"

_MODULE_HEADER: Final[str] = "\n".join(
    (
        "from __future__ import annotations",
        "import contextlib",
        "import gzip",
        "import hashlib",
        "import io",
        "import json",
        "import math",
        "import os",
        "import re",
        "import tempfile",
        "from collections import Counter, defaultdict",
        "from dataclasses import dataclass, field",
        "from pathlib import Path",
        "from typing import Any, Final, Mapping, Sequence",
        "import numpy as np",
        "import pandas as pd",
        "",
        "",
    )
)


class FrozenReferenceError(RuntimeError):
    """Raised when the frozen reference subset cannot be built."""


def frozen_script_dir() -> Path | None:
    """Directory holding the frozen scripts, or None when unset."""

    raw = os.environ.get(FROZEN_DIR_ENV_VAR)

    if not raw:
        return None

    directory = Path(raw).expanduser()

    if not directory.is_dir():
        return None

    return directory


def _top_level_bindings(
    tree: ast.Module,
) -> tuple[dict[str, ast.stmt], dict[str, ast.stmt]]:
    """Map module-level definition and assignment names to their nodes."""

    definitions: dict[str, ast.stmt] = {}
    assignments: dict[str, ast.stmt] = {}

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            definitions[node.name] = node

        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    assignments[target.id] = node

        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name) and node.value is not None:
                assignments[node.target.id] = node

    return definitions, assignments


def _referenced_names(node: ast.stmt) -> set[str]:
    return {
        child.id
        for child in ast.walk(node)
        if isinstance(child, ast.Name)
    }


def build_reference_module(
    script_path: Path,
    requested_names: tuple[str, ...],
    *,
    module_name: str | None = None,
) -> types.ModuleType:
    """
    Execute the requested names from ``script_path`` as a module.

    The transitive closure over module-level definitions and
    assignments is included, so a requested function arrives with the
    helpers and constants it depends on. Statements are emitted in
    their original source order.
    """

    if not script_path.is_file():
        raise FrozenReferenceError(
            f"Frozen script not found: {script_path}"
        )

    tree = ast.parse(
        script_path.read_text(encoding="utf-8"),
        filename=str(script_path),
    )

    definitions, assignments = _top_level_bindings(tree)
    available = set(definitions) | set(assignments)

    unknown = [
        name for name in requested_names if name not in available
    ]

    if unknown:
        raise FrozenReferenceError(
            f"{script_path.name} does not define: {sorted(unknown)}"
        )

    closure: set[str] = set(requested_names)
    frontier: list[str] = list(requested_names)

    while frontier:
        current = frontier.pop()
        node = definitions.get(current) or assignments.get(current)

        if node is None:
            continue

        for name in _referenced_names(node):
            if name in available and name not in closure:
                closure.add(name)
                frontier.append(name)

    selected = sorted(
        {
            id(node): node
            for name in closure
            if (node := definitions.get(name) or assignments.get(name))
            is not None
        }.values(),
        key=lambda node: node.lineno,
    )

    source = _MODULE_HEADER + "\n\n".join(
        ast.unparse(node) for node in selected
    )

    resolved_name = module_name or (
        f"frozen_reference_{script_path.stem}"
    )

    spec = importlib.util.spec_from_loader(resolved_name, loader=None)

    if spec is None:
        raise FrozenReferenceError(
            f"Could not create a module spec for {resolved_name}"
        )

    module = importlib.util.module_from_spec(spec)

    # Registered before execution because dataclass field resolution
    # looks the defining module up in sys.modules.
    sys.modules[resolved_name] = module

    try:
        exec(compile(source, f"<{resolved_name}>", "exec"), module.__dict__)
    except Exception as error:
        del sys.modules[resolved_name]

        raise FrozenReferenceError(
            f"Could not execute the reference subset of {script_path.name}"
        ) from error

    return module
