# SPDX-License-Identifier: AGPL-3.0-only

"""The multi-adapter merge method list is declared once per layer, and every
layer must agree.

`unsloth/multi_adapter_merge.py` is the source of truth. The CLI keeps a plain
literal because the core module imports torch and the CLI imports its backend
lazily, the Studio backend repeats the set in a pydantic ``Literal``, and the
frontend repeats it in ``MergeMethodType``/``MERGE_METHODS``. Each copy has
drifted at least once (see the `cat` and `dare_linear`/`magnitude_prune`
commits), so this test compares them all rather than trusting any one of them.

Parsed with ``ast``/regex instead of imports so the test runs without torch,
without the Studio backend, and without a Node toolchain.
"""

import ast
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

CORE = REPO_ROOT / "unsloth" / "multi_adapter_merge.py"
CLI = REPO_ROOT / "unsloth_cli" / "commands" / "export.py"
BACKEND_MODELS = REPO_ROOT / "studio" / "backend" / "models" / "export.py"
FRONTEND_CONSTANTS = (
    REPO_ROOT / "studio" / "frontend" / "src" / "features" / "export" / "constants.ts"
)


def _literal_tuple(path: Path, name: str) -> list:
    """Return the string elements of a module-level ``name = (...)`` assignment."""
    tree = ast.parse(path.read_text(encoding = "utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name for target in node.targets
        ):
            return [elt.value for elt in node.value.elts]
    raise AssertionError(f"{name} not found in {path}")


def _merge_request_literal(path: Path) -> list:
    """Return the string elements of ``MultiAdapterMergeRequest.method: Literal[...]``."""
    tree = ast.parse(path.read_text(encoding = "utf-8"))
    for cls in tree.body:
        if not (isinstance(cls, ast.ClassDef) and cls.name == "MultiAdapterMergeRequest"):
            continue
        for stmt in cls.body:
            if isinstance(stmt, ast.AnnAssign) and getattr(stmt.target, "id", None) == "method":
                return [elt.value for elt in stmt.annotation.slice.elts]
    raise AssertionError(f"MultiAdapterMergeRequest.method not found in {path}")


def _frontend_merge_methods(path: Path) -> list:
    """Return the ``value`` entries of the frontend ``MERGE_METHODS`` array."""
    source = path.read_text(encoding = "utf-8")
    body = re.search(r"export const MERGE_METHODS\b.*?= \[(.*?)\n\];", source, re.S)
    assert body is not None, f"MERGE_METHODS array not found in {path}"
    return re.findall(r'\{\s*value:\s*"([^"]+)"', body.group(1))


def _frontend_merge_method_type(path: Path) -> list:
    """Return the members of the frontend ``MergeMethodType`` union."""
    source = path.read_text(encoding = "utf-8")
    union = re.search(r"export type MergeMethodType =(.*?);", source, re.S)
    assert union is not None, f"MergeMethodType not found in {path}"
    return re.findall(r'\|\s*"([^"]+)"', union.group(1))


def test_core_declares_the_expected_methods():
    assert _literal_tuple(CORE, "SUPPORTED_METHODS") == [
        "linear",
        "ties",
        "dare_ties",
        "dare_linear",
        "magnitude_prune",
        "ctm",
        "cat",
    ]


def test_cli_lists_every_core_method():
    core = _literal_tuple(CORE, "SUPPORTED_METHODS")
    assert _literal_tuple(CLI, "MERGE_METHODS") == core, (
        "unsloth_cli/commands/export.py MERGE_METHODS drifted from "
        "SUPPORTED_METHODS in unsloth/multi_adapter_merge.py"
    )


def test_studio_request_accepts_every_core_method():
    core = _literal_tuple(CORE, "SUPPORTED_METHODS")
    assert _merge_request_literal(BACKEND_MODELS) == core, (
        "MultiAdapterMergeRequest.method drifted from SUPPORTED_METHODS in "
        "unsloth/multi_adapter_merge.py"
    )


def test_frontend_type_and_list_match_the_core_methods():
    core = _literal_tuple(CORE, "SUPPORTED_METHODS")
    assert _frontend_merge_method_type(FRONTEND_CONSTANTS) == core, (
        "MergeMethodType drifted from SUPPORTED_METHODS in "
        "unsloth/multi_adapter_merge.py"
    )
    assert _frontend_merge_methods(FRONTEND_CONSTANTS) == core, (
        "MERGE_METHODS drifted from SUPPORTED_METHODS in "
        "unsloth/multi_adapter_merge.py"
    )


def test_every_layer_names_the_same_set_once():
    """No layer may invent its own wording for the same method name."""
    layers = {
        "core": _literal_tuple(CORE, "SUPPORTED_METHODS"),
        "cli": _literal_tuple(CLI, "MERGE_METHODS"),
        "backend": _merge_request_literal(BACKEND_MODELS),
        "frontend_type": _frontend_merge_method_type(FRONTEND_CONSTANTS),
        "frontend_list": _frontend_merge_methods(FRONTEND_CONSTANTS),
    }
    first = layers["core"]
    for name, methods in layers.items():
        assert methods == first, f"{name} disagrees with core: {methods} != {first}"