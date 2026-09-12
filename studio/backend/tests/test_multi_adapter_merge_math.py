# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Math tests for the newer merge strategies in unsloth.multi_adapter_merge.

Loads the core module directly by file path (no package init), then checks each
per-module merge key against hand-computed expectations on tiny matrices:
dare_linear (drop & rescale + weighted sum), magnitude_prune (top-density
sparsify + weighted sum), and cat (factor concatenation == the weighted sum in
weight space, with negative-weight sign handling).
"""

import importlib.util
import unittest
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parents[3]


def _load_core_module():
    spec = importlib.util.spec_from_file_location(
        "multi_adapter_merge_under_test",
        _REPO_ROOT / "unsloth" / "multi_adapter_merge.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestNewMergeStrategies(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.core = _load_core_module()

    def test_new_methods_are_registered(self):
        for method in ("dare_linear", "magnitude_prune", "cat"):
            self.assertIn(method, self.core.SUPPORTED_METHODS)

    def test_dare_linear_with_zero_drop_is_the_weighted_sum(self):
        d1 = torch.tensor([[1.0, -2.0], [3.0, 0.5]])
        d2 = torch.tensor([[0.4, 0.8], [-1.0, 2.0]])
        merged = self.core._dare_linear_merge_key([d1, d2], [0.7, 0.3], drop_rate = 0.0)
        self.assertTrue(torch.allclose(merged, 0.7 * d1 + 0.3 * d2))

    def test_dare_linear_is_deterministic_for_a_seed(self):
        torch.manual_seed(1234)
        d1 = torch.randn(8, 8)
        d2 = torch.randn(8, 8)
        first = self.core._dare_linear_merge_key(
            [d1, d2], [0.5, 0.5], drop_rate = 0.5, seed = 7
        )
        second = self.core._dare_linear_merge_key(
            [d1, d2], [0.5, 0.5], drop_rate = 0.5, seed = 7
        )
        self.assertTrue(torch.equal(first, second))

    def test_dare_linear_rescales_surviving_deltas(self):
        # A zeroed partner isolates adapter 1's mask: with drop_rate=0.5 the
        # rescale factor is 1/(1 - 0.5) = 2, so surviving 0.5-weighted entries
        # land back at their original values.
        torch.manual_seed(9)
        d1 = torch.randn(8, 8)
        zeros = torch.zeros_like(d1)
        merged = self.core._dare_linear_merge_key(
            [d1, zeros], [0.5, 0.5], drop_rate = 0.5, seed = 7
        )
        nonzero = merged != 0
        self.assertTrue(nonzero.any())
        self.assertTrue(
            torch.allclose(merged[nonzero], d1[nonzero], atol = 1e-5)
        )

    def test_magnitude_prune_keeps_the_top_density_fraction(self):
        delta = torch.tensor([[-5.0, 0.1, 3.0, -0.2], [0.05, 4.0, -0.3, 2.0]])
        zeros = torch.zeros_like(delta)
        # density 0.5 of 8 elements keeps exactly the top 4 by |value|.
        merged = self.core._magnitude_prune_merge_key(
            [delta, zeros], [1.0, 1.0], density = 0.5
        )
        expected = torch.tensor([[-5.0, 0.0, 3.0, 0.0], [0.0, 4.0, 0.0, 2.0]])
        self.assertTrue(torch.equal(merged, expected))

    def test_magnitude_prune_with_full_density_is_the_weighted_sum(self):
        d1 = torch.randn(6, 6)
        d2 = torch.randn(6, 6)
        merged = self.core._magnitude_prune_merge_key(
            [d1, d2], [0.6, 0.4], density = 1.0
        )
        self.assertTrue(torch.allclose(merged, 0.6 * d1 + 0.4 * d2))

    def test_cat_matches_the_weighted_sum_in_weight_space(self):
        A1 = torch.randn(4, 8)
        B1 = torch.randn(6, 4)
        A2 = torch.randn(4, 8)
        B2 = torch.randn(6, 4)
        merged = self.core._cat_merge_key(
            [(A1, B1, 2.0, 0.7), (A2, B2, 1.0, 0.3)]
        )
        expected = 0.7 * 2.0 * B1 @ A1 + 0.3 * 1.0 * B2 @ A2
        self.assertTrue(torch.allclose(merged, expected, atol = 1e-4))

    def test_cat_flips_the_sign_of_negative_weights(self):
        A = torch.randn(4, 8)
        B = torch.randn(6, 4)
        merged = self.core._cat_merge_key([(A, B, 2.0, -0.5)])
        self.assertTrue(torch.allclose(merged, -0.5 * 2.0 * B @ A, atol = 1e-5))

    def test_single_adapter_shortcuts_skip_the_strategy_math(self):
        d = torch.randn(5, 5)
        self.assertTrue(
            torch.equal(
                self.core._dare_linear_merge_key([d], [0.7], drop_rate = 0.9),
                0.7 * d,
            )
        )
        self.assertTrue(
            torch.equal(
                self.core._magnitude_prune_merge_key([d], [0.7], density = 0.1),
                0.7 * d,
            )
        )


if __name__ == "__main__":
    unittest.main()