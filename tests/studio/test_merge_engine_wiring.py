# SPDX-License-Identifier: AGPL-3.0-only

"""Wiring contract for the mergekit merge engine.

The engine itself is covered in ``tests/saving/test_mergekit_bridge.py``; this
file only pins how the call sites are wired to it, using ``ast`` so it runs
without torch, without the Studio backend and without mergekit installed:

  * ``FastLanguageModel.merge_adapters`` takes ``engine`` and, on the mergekit
    path, merges to a temp dir and loads *that* back with ``from_pretrained``.
  * ``model.merge_multi_adapters`` forwards ``engine`` so an already-loaded
    model can explicitly reject the checkpoint-writing engine.
  * ``merge_adapters_into_model`` (the in-house engine) rejects
    ``engine="mergekit"`` **before** any adapter resolution or model work.
  * ``unsloth export merge-adapters`` exposes ``--engine``.
"""

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
LOADER = REPO_ROOT / "unsloth" / "models" / "loader.py"
SAVE = REPO_ROOT / "unsloth" / "save.py"
CORE = REPO_ROOT / "unsloth" / "multi_adapter_merge.py"
CLI = REPO_ROOT / "unsloth_cli" / "commands" / "export.py"


def _tree(path):
    return ast.parse(path.read_text(encoding = "utf-8"))


def _function(tree, name, cls_name = None):
    """Find a function, optionally inside one named class."""
    for node in ast.walk(tree):
        if cls_name is not None:
            if isinstance(node, ast.ClassDef) and node.name == cls_name:
                for item in node.body:
                    if isinstance(item, ast.FunctionDef) and item.name == name:
                        return item
            continue
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    return None


def _arg_names(fn):
    args = fn.args
    return [a.arg for a in list(args.args) + list(args.kwonlyargs)]


def _call_names(fn):
    """Names of every function called inside *fn* (attribute or bare)."""
    names = []
    for node in ast.walk(fn):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                names.append(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                names.append(node.func.attr)
    return names


def _string_literals(fn):
    return {
        node.value for node in ast.walk(fn)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }


def _keyword_names(fn, call_attr):
    """Keyword names of every call to *call_attr* (bare name or attribute)."""
    names = set()
    for node in ast.walk(fn):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        called = (
            func.attr if isinstance(func, ast.Attribute)
            else func.id if isinstance(func, ast.Name)
            else None
        )
        if called == call_attr:
            names.update(kw.arg for kw in node.keywords)
    return names


def test_loader_merge_adapters_exposes_engine():
    fn = _function(_tree(LOADER), "merge_adapters", cls_name = "FastLanguageModel")
    assert fn is not None, "FastLanguageModel.merge_adapters not found"
    assert "engine" in _arg_names(fn), "merge_adapters must accept engine"


def test_loader_mergekit_path_merges_to_a_temp_dir_then_loads_it():
    fn = _function(_tree(LOADER), "merge_adapters", cls_name = "FastLanguageModel")
    calls = _call_names(fn)
    assert "merge_adapters_via_mergekit" in calls
    assert "make_merge_output_dir" in calls, "merged checkpoint needs a temp dir"
    assert "normalize_out_dtype" in calls, "a torch dtype string would break mergekit"
    assert "resolve_engine" in calls

    bridge_kwargs = _keyword_names(fn, "merge_adapters_via_mergekit")
    for key in ("base_model", "adapters", "output_dir", "weights", "method", "out_dtype"):
        assert key in bridge_kwargs, f"mergekit call missing {key}"

    # The merged checkpoint is what gets loaded back, not the original name.
    from_pretrained = [
        node
        for node in ast.walk(fn)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "from_pretrained"
    ]
    assert from_pretrained, "mergekit path must load the merged checkpoint back"
    model_names = [
        kw.value.id
        for call in from_pretrained
        for kw in call.keywords
        if kw.arg == "model_name" and isinstance(kw.value, ast.Name)
    ]
    assert "merged_dir" in model_names, (
        "the mergekit path must pass the merged dir to from_pretrained; "
        f"got model_name sources {model_names}"
    )


def test_loader_still_delegates_to_the_in_house_engine():
    fn = _function(_tree(LOADER), "merge_adapters", cls_name = "FastLanguageModel")
    calls = _call_names(fn)
    assert "merge_adapters_into_model" in calls
    assert "resolve_engine" in calls


def test_save_wrapper_forwards_engine():
    source = SAVE.read_text(encoding = "utf-8")
    fn = _function(_tree(SAVE), "_unsloth_merge_multi_adapters")
    assert fn is not None, "_unsloth_merge_multi_adapters not found"
    assert "engine" in _arg_names(fn), "instance API must forward engine"
    assert "engine" in _keyword_names(fn, "merge_adapters_into_model")
    # Both monkeypatch sites keep pointing at the same wrapper.
    assert source.count("_unsloth_merge_multi_adapters, model") == 2
    # A global UNSLOTH_MERGE_ENGINE=mergekit must not break the in-memory API,
    # so an unset engine is pinned to legacy instead of read ambiently. Only the
    # executable statements count here - the docstring mentions the variable.
    executable = " ".join(
        segment
        for stmt in fn.body
        if not (isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant))
        for segment in sorted(
            node.value for node in ast.walk(stmt)
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
        )
    )
    assert "legacy" in executable
    assert "UNSLOTH_MERGE_ENGINE" not in executable


def test_core_rejects_mergekit_before_doing_any_work():
    fn = _function(_tree(CORE), "merge_adapters_into_model")
    assert fn is not None
    assert "engine" in _arg_names(fn)
    guard_at = None
    work_at = None
    for index, stmt in enumerate(fn.body):
        segment = ast.dump(stmt)
        if guard_at is None and "mergekit" in segment and isinstance(stmt, ast.If):
            guard_at = index
        if work_at is None and "MultiAdapterMergeConfig" in segment:
            work_at = index
    assert guard_at is not None, "engine='mergekit' guard not found"
    assert work_at is not None, "could not locate the first real work statement"
    assert guard_at < work_at, "the guard must run before the config is built"
    literals = " ".join(sorted(_string_literals(fn)))
    assert "writes a merged checkpoint" in literals


def test_cli_exposes_engine_flag_and_marks_mergekit_scope():
    fn = _function(_tree(CLI), "merge_adapters")
    assert fn is not None
    assert "engine" in _arg_names(fn)
    assert "engine" in _keyword_names(fn, "merge_adapters")
    literals = " ".join(sorted(_string_literals(fn)))
    assert "mergekit only implements" in literals
    assert "linear/ties/dare_ties/dare_linear" in literals