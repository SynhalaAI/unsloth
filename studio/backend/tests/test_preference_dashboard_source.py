# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Preference-objective UI guards: full-finetuning labels, the CPO chart card and the
Training Config popover's β / CPO α / max prompt rows.

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
