"""Import-direction guards, so the data <-> speller cycle cannot come back."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

PACKAGE = Path(__file__).resolve().parents[1] / "pattern_recognition"

# Each layer may only import from itself and the layers listed here.
ALLOWED: dict[str, set[str]] = {
    "selections": {"selections", "data"},  # data.splits only; asserted below
    "data": {"data", "selections"},
    "models": {"models"},
    # ``losses`` is a top-level leaf module (BrierLoss, GraphLoss).
    "training": {"training", "models", "data", "selections", "losses"},
    "experiment": {
        "experiment",
        "data",
        "models",
        "training",
        "selections",
        "losses",
    },
    "reporting": {"reporting"},
    "speller": {
        "speller",
        "selections",
        "data",
        "models",
        "training",
        "experiment",
        "reporting",
    },
}


def _imported_layers(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(), filename=str(path))
    layers: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            module = node.module
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("pattern_recognition."):
                    layers.add(alias.name.split(".")[1])
            continue
        else:
            continue
        if module.startswith("pattern_recognition."):
            layers.add(module.split(".")[1])
    return layers


def _modules_in(layer: str) -> list[Path]:
    return sorted((PACKAGE / layer).rglob("*.py"))


@pytest.mark.parametrize("layer", sorted(ALLOWED))
def test_layer_only_imports_allowed_layers(layer):
    allowed = ALLOWED[layer]
    violations = []
    for path in _modules_in(layer):
        for imported in _imported_layers(path) - allowed:
            violations.append(f"{path.relative_to(PACKAGE)} imports {imported}")
    assert not violations, "backwards imports:\n" + "\n".join(violations)


def test_selections_is_a_leaf_apart_from_splits():
    """selections/ is shared by both sides, so it must stay near-dependency-free."""
    for path in _modules_in("selections"):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                if node.module.startswith(
                    "pattern_recognition."
                ) and not node.module.startswith("pattern_recognition.selections"):
                    assert node.module == "pattern_recognition.data.splits", (
                        f"{path.relative_to(PACKAGE)} imports {node.module}; "
                        "selections/ may only reach data.splits"
                    )


def test_pipelines_have_no_cycle_dodging_lazy_imports():
    """Function-level package imports were workarounds; the cycle is gone now."""
    lazy = []
    for path in _modules_in("data"):
        tree = ast.parse(path.read_text(), filename=str(path))
        for func in ast.walk(tree):
            if not isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for node in ast.walk(func):
                if isinstance(node, ast.ImportFrom) and (node.module or "").startswith(
                    "pattern_recognition"
                ):
                    lazy.append(f"{path.relative_to(PACKAGE)}:{node.lineno}")
    assert not lazy, "function-level package imports left in data/: " + ", ".join(lazy)


def test_protocols_are_registered_by_importing_selections():
    """Registration must not depend on the speller package being imported."""
    from pattern_recognition.selections import get_protocol

    assert get_protocol("bci3_rowcol").name == "bci3_rowcol"
    assert get_protocol("samara_single_flash_sim").name == "samara_single_flash_sim"
