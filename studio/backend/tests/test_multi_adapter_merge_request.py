# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Schema tests for MultiAdapterMergeRequest (the Export page merge config).

The API schema is the gate in front of ``unsloth.multi_adapter_merge``: every method in
the core's SUPPORTED_METHODS tuple must be accepted here, and the method-specific knobs
(density, target_rank) must stay inside the ranges the core validates.
"""

import importlib.util
import unittest
from pathlib import Path

_BACKEND_ROOT = Path(__file__).resolve().parent.parent


def _load_models_module():
    spec = importlib.util.spec_from_file_location(
        "export_models_for_merge_request_test",
        _BACKEND_ROOT / "models" / "export.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestMultiAdapterMergeRequest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.models = _load_models_module()
        cls.schema = cls.models.MultiAdapterMergeRequest

    def test_all_core_merge_methods_are_accepted(self):
        # Keep in sync with SUPPORTED_METHODS in unsloth/multi_adapter_merge.py.
        for method in (
            "linear", "svd", "cat", "ties", "dare_ties", "dare_linear",
            "magnitude_prune",
            "ties_svd", "dare_ties_svd", "dare_linear_svd",
            "magnitude_prune_svd",
        ):
            model = self.schema(
                adapter_paths = ["a", "b"], weights = [1.0, 1.0], method = method
            )
            self.assertEqual(model.method, method)

    def test_unknown_method_is_rejected(self):
        # The core accepts "dare"/"ctm" aliases; the API stays strict on canonical names,
        # and the removed in-house methods are gone entirely.
        for method in ("dare", "ctm", "sce", "della", "model_stock", "bogus"):
            with self.assertRaises(Exception):
                self.schema(adapter_paths = ["a", "b"], method = method)

    def test_defaults_match_the_core_merger(self):
        model = self.schema(adapter_paths = ["a", "b"])
        self.assertEqual(model.method, "linear")
        self.assertEqual(model.density, 0.5)
        self.assertIsNone(model.target_rank)
        self.assertTrue(model.normalize_weights)

    def test_density_must_stay_in_range(self):
        ok = self.schema(adapter_paths = ["a", "b"], method = "ties", density = 0.25)
        self.assertEqual(ok.density, 0.25)
        for bad in (0.0, -0.1, 1.5):
            with self.assertRaises(Exception):
                self.schema(adapter_paths = ["a", "b"], method = "ties", density = bad)

    def test_target_rank_must_be_positive(self):
        ok = self.schema(adapter_paths = ["a", "b"], method = "svd", target_rank = 16)
        self.assertEqual(ok.target_rank, 16)
        for bad in (0, -4):
            with self.assertRaises(Exception):
                self.schema(adapter_paths = ["a", "b"], method = "svd", target_rank = bad)

    def test_weights_must_match_adapter_count(self):
        with self.assertRaises(Exception):
            self.schema(adapter_paths = ["a", "b"], weights = [1.0])

    def test_a_single_adapter_is_a_valid_merge(self):
        # Base + one LoRA is the degenerate merge: the schema allows a single path.
        model = self.schema(adapter_paths = ["only"])
        self.assertEqual(model.adapter_paths, ["only"])
        self.assertIsNone(model.weights)
        self.assertEqual(model.method, "linear")
        # The weight list, when given, still has to line up.
        ok = self.schema(adapter_paths = ["only"], weights = [0.75])
        self.assertEqual(ok.weights, [0.75])
        with self.assertRaises(Exception):
            self.schema(adapter_paths = ["only"], weights = [1.0, 1.0])

    def test_every_method_accepts_a_single_adapter(self):
        # PEFT falls back to linear for one adapter, so no method needs more.
        for method in (
            "linear", "svd", "cat", "ties", "dare_ties", "dare_linear",
            "magnitude_prune",
            "ties_svd", "dare_ties_svd", "dare_linear_svd",
            "magnitude_prune_svd",
        ):
            model = self.schema(adapter_paths = ["only"], method = method)
            self.assertEqual(model.method, method)


if __name__ == "__main__":
    unittest.main()