# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Unit tests for the Gemma 4 axial vision-RoPE compatibility shim (CPU-only)."""

from __future__ import annotations

from utils.gemma4_axial_compat import register_axial_rope, register_axial_validation


class _LegacyVisionRope:
    """transformers 5.5/5.10: "default" is the axial 2D RoPE."""

    @staticmethod
    def compute_default_rope_parameters(config, device = None, seq_len = None):
        return "axial_inv_freq", 1.0


class _NativeAxialVisionRope:
    """transformers >= 5.20: axial is resolved without ROPE_INIT_FUNCTIONS."""

    @staticmethod
    def compute_axial_rope_parameters(config, device = None, **kwargs):
        return "axial_inv_freq", 1.0


class _LegacyVisionConfig:
    """transformers 5.5/5.10: only the default 2D validator exists."""

    def __init__(self):
        self.validated = []

    def _validate_default_rope_parameters(self, rope_parameters, ignore_keys = None):
        self.validated.append(dict(rope_parameters))


def test_registers_axial_for_legacy_transformers():
    rope_init_functions = {"default": object()}
    assert register_axial_rope(rope_init_functions, _LegacyVisionRope) is True
    assert rope_init_functions["axial"] is _LegacyVisionRope.compute_default_rope_parameters


def test_keeps_an_existing_axial_entry():
    existing = object()
    rope_init_functions = {"axial": existing}
    assert register_axial_rope(rope_init_functions, _LegacyVisionRope) is False
    assert rope_init_functions["axial"] is existing


def test_does_not_override_native_axial_support():
    rope_init_functions = {}
    assert register_axial_rope(rope_init_functions, _NativeAxialVisionRope) is False
    assert "axial" not in rope_init_functions


def test_class_without_default_rope_is_left_alone():
    class _Unknown:
        pass

    rope_init_functions = {}
    assert register_axial_rope(rope_init_functions, _Unknown) is False
    assert rope_init_functions == {}


def test_registers_the_axial_validator_for_legacy_transformers():
    assert register_axial_validation(_LegacyVisionConfig) is True
    config = _LegacyVisionConfig()
    config._validate_axial_rope_parameters({"rope_type": "axial", "rope_theta": 100.0})
    # Axial validates like the default rope, so the required-key checks still run.
    assert config.validated == [{"rope_type": "axial", "rope_theta": 100.0}]


def test_does_not_override_native_axial_validation():
    class _NativeVisionConfig:
        def _validate_axial_rope_parameters(self, rope_parameters, ignore_keys = None):
            raise AssertionError("must not be replaced")

        def _validate_default_rope_parameters(self, rope_parameters, ignore_keys = None):
            raise AssertionError("must not be aliased")

    assert register_axial_validation(_NativeVisionConfig) is False


def test_config_without_default_validation_is_left_alone():
    class _Unknown:
        pass

    assert register_axial_validation(_Unknown) is False
    assert not hasattr(_Unknown, "_validate_axial_rope_parameters")
