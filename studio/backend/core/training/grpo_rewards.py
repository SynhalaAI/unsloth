# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Built-in verifiable reward functions for Studio GRPO training.

All reward functions follow TRL ``GRPOTrainer``'s signature ``(prompts, completions, **kwargs)``
and support both plain-string completions and conversational completions
(``[{"role": "assistant", "content": "..."}]``) emitted when the dataset ``prompt`` is conversational.
"""

from __future__ import annotations

import re
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

GRPO_REWARD_PRESETS = (
    "answer_match",
    "answer_format",
    "reasoning_format",
    "conciseness",
)

DEFAULT_GRPO_REWARD_FUNCS = ("answer_match", "answer_format")

_ANSWER_TAG_RE = re.compile(r"<answer>(.*?)</answer>", re.IGNORECASE | re.DOTALL)
_REASONING_TAG_RE = re.compile(
    r"<(?:think|reasoning)>(.*?)</(?:think|reasoning)>", re.IGNORECASE | re.DOTALL
)
_HASH_ANSWER_RE = re.compile(r"####\s*([^\n]+)")
_TARGET_KWARG_KEYS = (
    "answer",
    "solution",
    "ground_truth",
    "expected_answer",
    "final_answer",
    "target",
    "reference",
)


def extract_completion_text(completion: Any) -> str:
    """Normalize a TRL completion (plain string or conversational message list) to text."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, dict):
        return str(completion.get("content", completion.get("value", "")))
    if isinstance(completion, (list, tuple)):
        parts: List[str] = []
        for item in completion:
            if isinstance(item, dict):
                content = item.get("content", item.get("value", ""))
                if content is not None:
                    parts.append(str(content))
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts)
    return str(completion or "")


def _extract_boxed(text: str) -> Optional[str]:
    marker = r"\boxed{"
    idx = text.rfind(marker)
    if idx == -1:
        return None
    start = idx + len(marker)
    depth = 1
    pos = start
    while pos < len(text) and depth > 0:
        ch = text[pos]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
        pos += 1
    if depth == 0:
        return text[start : pos - 1].strip()
    return None


def extract_answer(text: str) -> str:
    """Extract the final answer from ``<answer>...</answer>``, ``\\boxed{...}``, or ``#### ...``."""
    tag_matches = _ANSWER_TAG_RE.findall(text)
    if tag_matches:
        return tag_matches[-1].strip()
    boxed = _extract_boxed(text)
    if boxed is not None:
        return boxed
    hash_matches = _HASH_ANSWER_RE.findall(text)
    if hash_matches:
        return hash_matches[-1].strip()
    return text.strip()


def _normalize_answer(value: Any) -> str:
    text = str(value or "").strip()
    text = re.sub(r"^\(([^()]+)\)$", r"\1", text).strip()
    text = re.sub(r"^([A-Za-z0-9]+)[).:]$", r"\1", text).strip()
    text = re.sub(r"\s+", " ", text)
    return text.casefold()


def _resolve_targets(count: int, kwargs: Dict[str, Any]) -> Optional[Sequence[Any]]:
    for key in _TARGET_KWARG_KEYS:
        values = kwargs.get(key)
        if isinstance(values, (list, tuple)) and len(values) == count:
            return values
    return None


def answer_match_reward(
    prompts: Any = None,
    completions: Sequence[Any] = (),
    **kwargs: Any,
) -> List[float]:
    """Reward 1.0 when the extracted completion answer matches the ground-truth target, else 0.0."""
    del prompts
    targets = _resolve_targets(len(completions), kwargs)
    if targets is None:
        return [0.0 for _ in completions]

    rewards: List[float] = []
    for completion, target in zip(completions, targets):
        text = extract_completion_text(completion)
        predicted = _normalize_answer(extract_answer(text))
        expected = _normalize_answer(extract_answer(str(target)))
        if predicted and expected and predicted == expected:
            rewards.append(1.0)
        else:
            rewards.append(0.0)
    return rewards


def answer_format_reward(
    prompts: Any = None,
    completions: Sequence[Any] = (),
    **kwargs: Any,
) -> List[float]:
    """Reward valid ``<answer>...</answer>`` tags (0.5 full match, 0.35 repeated, 0.1 partial, 0.0 missing)."""
    del prompts, kwargs
    rewards: List[float] = []
    for completion in completions:
        text = extract_completion_text(completion)
        matches = _ANSWER_TAG_RE.findall(text)
        if matches and matches[-1].strip():
            rewards.append(0.5 if len(matches) == 1 else 0.35)
        elif "<answer>" in text.lower() or "</answer>" in text.lower():
            rewards.append(0.1)
        else:
            rewards.append(0.0)
    return rewards


def reasoning_format_reward(
    prompts: Any = None,
    completions: Sequence[Any] = (),
    **kwargs: Any,
) -> List[float]:
    """Reward structured ``<think>...</think>`` (or ``<reasoning>...</reasoning>``) + ``<answer>``."""
    del prompts, kwargs
    rewards: List[float] = []
    for completion in completions:
        text = extract_completion_text(completion)
        has_reasoning = bool(_REASONING_TAG_RE.search(text))
        has_answer = bool(_ANSWER_TAG_RE.search(text))
        if has_reasoning and has_answer:
            rewards.append(0.5)
        elif has_reasoning or has_answer:
            rewards.append(0.2)
        else:
            rewards.append(0.0)
    return rewards


def conciseness_reward(
    prompts: Any = None,
    completions: Sequence[Any] = (),
    **kwargs: Any,
) -> List[float]:
    """Reward non-empty completions without runaway repetition."""
    del prompts, kwargs
    rewards: List[float] = []
    for completion in completions:
        text = extract_completion_text(completion).strip()
        if not text:
            rewards.append(0.0)
            continue
        words = text.split()
        if not words:
            rewards.append(0.0)
            continue
        unique_ratio = len(set(words)) / len(words)
        rewards.append(round(0.25 * min(1.0, unique_ratio * 1.25), 4))
    return rewards


_REWARD_REGISTRY: Dict[str, Callable[..., List[float]]] = {
    "answer_match": answer_match_reward,
    "answer_format": answer_format_reward,
    "reasoning_format": reasoning_format_reward,
    "conciseness": conciseness_reward,
}


def resolve_grpo_reward_funcs(
    reward_names: Optional[Sequence[str]] = None,
    reward_weights: Optional[Sequence[float]] = None,
) -> Tuple[List[Callable[..., List[float]]], Optional[List[float]]]:
    """Resolve preset names into TRL-compatible reward callables and optional weights."""
    names = [
        str(name).strip().lower()
        for name in (reward_names or DEFAULT_GRPO_REWARD_FUNCS)
        if str(name).strip()
    ]
    if not names:
        names = list(DEFAULT_GRPO_REWARD_FUNCS)

    funcs: List[Callable[..., List[float]]] = []
    for name in names:
        fn = _REWARD_REGISTRY.get(name)
        if fn is None:
            raise ValueError(
                f"Unknown GRPO reward function {name!r}; expected one of {', '.join(GRPO_REWARD_PRESETS)}."
            )
        funcs.append(fn)

    weights: Optional[List[float]] = None
    if reward_weights is not None:
        weights = [float(w) for w in reward_weights]
        if len(weights) != len(funcs):
            raise ValueError(
                f"grpo_reward_weights length ({len(weights)}) must match grpo_reward_funcs ({len(funcs)})."
            )

    return funcs, weights