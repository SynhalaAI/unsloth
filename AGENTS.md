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

## Adapter merging — PEFT parity rule

`unsloth/multi_adapter_merge.py` implements **no merge arithmetic**. Every
method it exposes is a PEFT combination type, applied through
`peft.tuners.lora.model.LoraModel.add_weighted_adapter` and then merged into the
base weights with `merge_and_unload`.

| Method | PEFT `combination_type` | Reference |
|---|---|---|
| `linear` | `linear` | [PEFT `add_weighted_adapter`](https://github.com/huggingface/peft/blob/main/src/peft/tuners/lora/model.py) |
| `svd` | `svd`, with `svd_rank` from `target_rank` | same |
| `cat` | `cat` | same |
| `ties` | `ties` | [PEFT `merge_utils.ties`](https://github.com/huggingface/peft/blob/main/src/peft/utils/merge_utils.py) |
| `dare_ties` | `dare_ties` | [PEFT `merge_utils.dare_ties`](https://github.com/huggingface/peft/blob/main/src/peft/utils/merge_utils.py) |
| `dare_linear` | `dare_linear` | [PEFT `merge_utils.dare_linear`](https://github.com/huggingface/peft/blob/main/src/peft/utils/merge_utils.py) |
| `magnitude_prune` | `magnitude_prune` | [PEFT `merge_utils.magnitude_prune`](https://github.com/huggingface/peft/blob/main/src/peft/utils/merge_utils.py) |

The adapter I/O helpers (`_load_adapter_state_dict`, `_group_lora_factors`,
`_reconstruct_deltas`) stay in that module: the Studio interference preflight
(`studio/backend/core/export/merge_metrics.py`) reads adapters with them without
loading peft, so the module must keep importing nothing but torch and the
standard library at module level.

### Rules for merging-related changes

1. **Never** reimplement merge math in this repository. Fix or extend the
   algorithm in PEFT, and raise the `peft` floor in `pyproject.toml` when a fix
   is only needed by a newer PEFT.
2. `SUPPORTED_METHODS` is exactly the set of PEFT combination types Unsloth
   exposes. Add a name there only together with its mapping in
   `_peft_combination_kwargs`.
3. `density` is PEFT's keep fraction and `target_rank` is PEFT's `svd_rank`.
   Neither may be reinterpreted, and no method gets a second, private knob.
4. A test may compare a merge against PEFT's own output or against the
   reconstructed `ΔW = (α/r)·B·A` of an adapter - never against a mergekit
   reimplementation.
5. PEFT requires every adapter to share one LoRA rank for `linear`, `ties`,
   `dare_*` and `magnitude_prune`; `svd` and `cat` accept mixed ranks. Keep
   that constraint visible to callers instead of working around it silently.
6. PEFT can only union set-valued `target_modules`, so the core normalises each
   loaded adapter to a set of the modules it actually reached
   (`_normalize_peft_target_modules`) before combining. Keep that step when
   adapters are re-laid onto the model: a checkpoint saved with an explicit list
   and one saved as `"all-linear"` or a regex inject the same layers but PEFT
   refuses the mix otherwise.
7. Record those targets as exact module keys, never as leaf names. PEFT matches
   a set entry by exact key first and by suffix afterwards, so `"q_proj"` would
   pull same-named modules out of towers no adapter reached, and PEFT then
   combines an empty list there and raises `IndexError: list index out of
   range`. Multimodal checkpoints hit this whenever towers share projection
   names.

## When uncertain

If implementing the feature requires a large core rewrite, broad formatting changes,
or changes to public APIs, stop and explain the conflict risk before proceeding.
Propose the smallest isolated alternative first.
