# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Preference-objective UI guards: full-finetuning labels, the preference chart cards and
the Training Config popover's β / CPO α / max prompt rows.

These pin the frontend source because the dashboard is plain TSX; the behavioural
counterpart lives in ``frontend/tests/orpo-dashboard.test.ts``.
"""

from __future__ import annotations

import re
from pathlib import Path

_FRONTEND = Path(__file__).resolve().parents[2] / "frontend" / "src"
_STUDIO = _FRONTEND / "features" / "studio"


def _read(path: Path) -> str:
    return path.read_text(encoding = "utf-8")


def test_full_finetuning_preference_runs_get_the_objective_label():
    src = _read(_FRONTEND / "features" / "training" / "lib" / "training-methods.ts")
    # The backend dispatches full-finetuning runs to the same DPO/ORPO/CPO trainers, so
    # the label gate must be the shared objective check, not the adapter-only one.
    assert "!usesPreferenceObjective(trainingMethod, preferenceMethod)" in src
    assert 'trainingMethod !== "qlora" && trainingMethod !== "lora"' not in src


def test_wizard_preview_uses_the_shared_preference_gate():
    src = _read(_STUDIO / "wizard" / "run-preview-card.tsx")
    assert "usesPreferenceObjective(\n    trainingMethod," in src
    assert "isAdapterMethod(trainingMethod) &&" not in src


def test_orpo_components_card_needs_a_log_odds_ratio():
    src = _read(_STUDIO / "sections" / "charts-content.tsx")
    assert "Number.isFinite(point.logOddsRatio)" in src
    assert "{hasOrpoComponents && (" in src
    # CPO logs nll_loss only, so gating on nll_loss alone left a blank log-odds line.
    assert "{reducedOrpoData.length > 0 && (" not in src


def test_training_config_popover_shows_preference_hyperparams():
    override = _read(_STUDIO / "sections" / "run-config-override.ts")
    for key in ("config.dpo_beta", "config.cpo_alpha", "config.max_prompt_length"):
        assert key in override, f"run-config-override.ts must map {key}"
    src = _read(_STUDIO / "sections" / "progress-section.tsx")
    for key in (
        "studio.progress.preferenceBeta",
        "studio.progress.preferenceAlpha",
        "studio.progress.preferenceMaxPromptLength",
    ):
        assert key in src, f"progress-section.tsx must render {key}"
    assert "usesPreferenceObjective(data.trainingMethod, cfgPreferenceMethod)" in src


def test_every_locale_translates_the_new_popover_keys():
    locales = sorted((_FRONTEND / "i18n" / "locales").glob("*.ts"))
    assert len(locales) >= 14
    keys = (
        "preference",
        "preferenceBeta",
        "preferenceAlpha",
        "preferenceMaxPromptLength",
    )
    for path in locales:
        src = _read(path)
        for key in keys:
            hits = re.findall(r"(?m)^      %s: " % re.escape(key), src)
            assert len(hits) == 1, "%s must define studio.progress.%s once" % (
                path.name,
                key,
            )


def test_preference_config_import_export_keeps_the_objective():
    yaml_config = _read(
        _FRONTEND / "features" / "training" / "lib" / "yaml-config.ts"
    )
    defaults = _read(
        _FRONTEND / "features" / "training" / "lib" / "model-defaults.ts"
    )
    for key in ("training_method", "dpo_beta", "cpo_alpha", "max_prompt_length"):
        assert key in yaml_config, "yaml-config.ts must save %s" % key
        assert key in defaults, "model-defaults.ts must import %s" % key
    # SFT files carry the method too, so an import can clear a stale preference selection.
    assert "state.preferenceTrainingMethod.toUpperCase()" in yaml_config


def test_max_prompt_length_row_shows_the_trainers_cap():
    limits = _read(
        _FRONTEND / "features" / "training" / "lib" / "preference-limits.ts"
    )
    trainer = _read(
        _FRONTEND.parents[1]
        / "backend"
        / "core"
        / "training"
        / "preference_trainer.py"
    )
    section = _read(_STUDIO / "sections" / "training-hyperparameters-section.tsx")
    # The UI mirrors the trainer's own cap instead of offering values it would reduce.
    assert "max_length // 2" in trainer, "the trainer must keep its half-context cap"
    assert "export function maxPromptLengthCap" in limits
    assert re.search(
        r"effectiveMaxPromptLength\(\s*store\.contextLength,\s*store\.maxPromptLength,",
        section,
    ), "the row must show the capped value"
    assert "maxPromptLengthPresets(store.contextLength)" in section
    assert "MAX_PROMPT_LENGTH_OPTIONS" not in section


def test_preference_rows_are_hidden_for_cpt():
    section = _read(_STUDIO / "sections" / "training-hyperparameters-section.tsx")
    # CPT dispatches before the preference branch, so its knobs cannot change the run.
    assert "const showPreferenceRows =" in section
    assert 'store.preferenceTrainingMethod !== "sft" && !store.isCpt' in section
    assert "{showPreferenceRows && (" in section


def test_reward_level_chart_is_wired_end_to_end():
    card = _read(_STUDIO / "sections" / "charts" / "reward-chart-card.tsx")
    assert "export function RewardChartCard" in card
    assert 'data-tour="studio-training-reward"' in card
    assert 'label: "Reward (chosen)"' in card
    assert 'label: "Reward (rejected)"' in card

    content = _read(_STUDIO / "sections" / "charts-content.tsx")
    assert "const reducedRewardData = useMemo(" in content
    assert "<RewardChartCard data={reducedRewardData} />" in content

    section = _read(_STUDIO / "sections" / "charts-section.tsx")
    for prop in ("rewardsChosenHistory", "rewardsRejectedHistory"):
        assert prop in section, "charts-section.tsx must accept %s" % prop
    assert "rewardsChosenHistory: rewardsChosenHistory.map(" in section
    assert "rewardsRejectedHistory: rewardsRejectedHistory.map(" in section

    store = _read(
        _FRONTEND / "features" / "training" / "stores" / "training-runtime-store.ts"
    )
    for field in ("rewards_chosen", "rewards_rejected"):
        assert "history.%s && history.%s_steps" % (field, field) in store
    assert "upsertPoint(state.rewardsChosenHistory" in store
    assert "upsertPoint(state.rewardsRejectedHistory" in store

    # SSE and saved-run history must both reach the chart.
    for name in ("live-training-view.tsx", "historical-training-view.tsx"):
        view = _read(_STUDIO / name)
        assert "rewardsChosenHistory={viewData.rewardsChosenHistory}" in view
        assert "rewardsRejectedHistory={viewData.rewardsRejectedHistory}" in view

    # Passing the props is not enough: the live view builds viewData from its own useShallow
    # selector, so a missing key left both series undefined and the card empty.
    live = _read(_STUDIO / "live-training-view.tsx")
    for field in ("rewardsChosenHistory", "rewardsRejectedHistory"):
        assert "state.%s," % field in live, (
            "live-training-view.tsx must select state.%s" % field
        )
