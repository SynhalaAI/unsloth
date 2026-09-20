# Repository Editing Guidelines

## Goal

Implement features in a way that minimizes future merge conflicts with upstream
and with work from other contributors.

## Required workflow

- Before starting work, read this file and report that you have done so.
- Inspect the existing architecture and extension points before editing code.
- Prefer adding isolated modules over rewriting existing core modules.
- Keep changes narrowly scoped to the requested feature.
- Do not perform unrelated refactors, renames, formatting, or import reordering.
- Avoid modifying generated files, vendored code, lock files, and build artifacts
  unless the task explicitly requires it.
- Reuse existing hooks, registries, adapters, configuration mechanisms, and public
  interfaces whenever possible.
- Preserve backward compatibility unless the task explicitly authorizes a breaking
  change.
- If a frequently modified core file must be changed, keep the integration patch
  as small as possible and place the main implementation in a separate module.
- Do not overwrite or revert changes that are unrelated to the current task.
- Keep logical changes separated so they can be reviewed or reverted independently.
- Add or update focused tests for changed behavior without rewriting unrelated tests.
- Always include automated unit or integration tests for newly added features if they are missing or not yet covered.
- Before finishing, inspect the final diff, remove unrelated changes, and report
  that no unrelated diff remains.

## Adapter merging methods — reference parity rule

The in-house adapter merging implementations in `unsloth/multi_adapter_merge.py`
do **not** use mergekit as a dependency. All methods are re-implemented in pure
PyTorch, but their algorithms **must remain faithful to the published references**.

### Canonical references

| Method | Reference |
|---|---|
| `linear` | [mergekit `linear.py`](https://github.com/arcee-ai/mergekit/blob/main/mergekit/merge_methods/linear.py) |
| `ties` | [mergekit `generalized_task_arithmetic.py`](https://github.com/arcee-ai/mergekit/blob/main/mergekit/merge_methods/generalized_task_arithmetic.py) (``sum`` consensus, weight divisor) |
| `dare_ties` | [DARE paper (Yu et al., 2024)](https://arxiv.org/abs/2311.03099) + TIES |
| `dare_linear` | [DARE paper](https://arxiv.org/abs/2311.03099) + linear weighted sum |
| `magnitude_prune` | [PEFT `merge_utils.py`](https://github.com/huggingface/peft/blob/main/src/peft/utils/merge_utils.py) |
| `ctm` | Unsloth-specific (truncated-SVD compression); no external reference |
| `cat` | [PEFT `add_weighted_adapter` `combination_type="cat"`](https://github.com/huggingface/peft/blob/main/src/peft/tuners/lora/model.py) |
| `sce` | [mergekit `sce.py`](https://github.com/arcee-ai/mergekit/blob/main/mergekit/merge_methods/sce.py) ([SCE paper](https://arxiv.org/abs/2408.07990)) |
| `della` / `della_linear` | [mergekit `sparsify.py` `della_magprune`](https://github.com/arcee-ai/mergekit/blob/main/mergekit/sparsify.py) ([DELLA paper](https://arxiv.org/abs/2406.11617)) |
| `breadcrumbs` / `breadcrumbs_ties` | [mergekit `sparsify.py` `magnitude_outliers`](https://github.com/arcee-ai/mergekit/blob/main/mergekit/sparsify.py) ([Breadcrumbs paper](https://arxiv.org/abs/2312.06795)) |
| `multislerp` | [mergekit `multislerp.py`](https://github.com/arcee-ai/mergekit/blob/main/mergekit/merge_methods/multislerp.py) (delta-space adaptation; no base tensor) |
| `model_stock` | [mergekit `model_stock.py`](https://github.com/arcee-ai/mergekit/blob/main/mergekit/merge_methods/model_stock.py) ([Model Stock paper](https://arxiv.org/abs/2403.19522); delta-space adaptation, ≥3 adapters) |

### Rules for merging-related changes

1. **Never** modify merge-method math (trim, sign election, rescale, pruning,
   concatenation) without cross-checking against the canonical reference above.
2. If an upstream fix (mergekit, PEFT, or the original paper) changes an
   algorithm, update the in-house implementation **and** its tests in the same
   PR, and note the sync in the commit message.
3. New merge methods must cite their reference implementation or paper in the
   docstring.
4. The weekly `mergekit-parity-check` GitHub Actions workflow monitors
   `mergekit/merge_methods/` for upstream changes; act on issues it creates.

## When uncertain

If implementing the feature requires a large core rewrite, broad formatting changes,
or changes to public APIs, stop and explain the conflict risk before proceeding.
Propose the smallest isolated alternative first.
