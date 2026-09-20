# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Schema tests for MultiAdapterMergeRequest (the Export page merge config).

The API schema is the gate in front of ``unsloth.multi_adapter_merge``: every method in
the core's SUPPORTED_METHODS tuple must be accepted here, and the method-specific knobs
(density, drop_rate, target_rank) must stay inside the ranges the core validates.
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
            "linear", "ties", "dare_ties", "dare_linear",
            "magnitude_prune", "ctm", "cat", "sce", "della", "della_linear",
            "breadcrumbs", "breadcrumbs_ties", "multislerp", "model_stock",
        ):
            paths = ["a", "b", "c"] if method == "model_stock" else ["a", "b"]
            model = self.schema(
                adapter_paths = paths, weights = [1.0] * len(paths), method = method
            )
            self.assertEqual(model.method, method)

    def test_unknown_method_is_rejected(self):
        # The core accepts "dare"/"svd" aliases; the API stays strict on canonical names.
        for method in ("dare", "svd", "bogus"):
            with self.assertRaises(Exception):
                self.schema(adapter_paths = ["a", "b"], method = method)

    def test_defaults_match_the_core_merger(self):
        model = self.schema(adapter_paths = ["a", "b"])
        self.assertEqual(model.method, "linear")
        self.assertEqual(model.density, 0.5)
        self.assertEqual(model.drop_rate, 0.5)
        self.assertIsNone(model.target_rank)
        self.assertTrue(model.normalize_weights)

    def test_drop_rate_must_stay_in_the_dare_range(self):
        for method in ("dare_ties", "dare_linear"):
            ok = self.schema(adapter_paths = ["a", "b"], method = method, drop_rate = 0.0)
            self.assertEqual(ok.drop_rate, 0.0)
            for bad in (-0.1, 1.0, 2.0):
                with self.assertRaises(Exception):
                    self.schema(
                        adapter_paths = ["a", "b"], method = method, drop_rate = bad
                    )

    def test_target_rank_must_be_positive(self):
        ok = self.schema(adapter_paths = ["a", "b"], method = "ctm", target_rank = 16)
        self.assertEqual(ok.target_rank, 16)
        for bad in (0, -4):
            with self.assertRaises(Exception):
                self.schema(adapter_paths = ["a", "b"], method = "ctm", target_rank = bad)

    def test_weights_must_match_adapter_count(self):
        with self.assertRaises(Exception):
            self.schema(adapter_paths = ["a", "b"], weights = [1.0])

    def test_model_stock_requires_three_adapters(self):
        ok = self.schema(
            adapter_paths = ["a", "b", "c"], method = "model_stock"
        )
        self.assertEqual(ok.method, "model_stock")
        with self.assertRaises(Exception):
            self.schema(adapter_paths = ["a", "b"], method = "model_stock")

    def test_della_breadcrumbs_sce_knob_ranges(self):
        # DELLA epsilon must stay in (0, 1).
        ok = self.schema(
            adapter_paths = ["a", "b"], method = "della", della_epsilon = 0.2
        )
        self.assertEqual(ok.della_epsilon, 0.2)
        for bad in (0.0, -0.1, 1.0):
            with self.assertRaises(Exception):
                self.schema(
                    adapter_paths = ["a", "b"], method = "della", della_epsilon = bad
                )
        # Breadcrumbs gamma must stay in [0, 1).
        ok = self.schema(
            adapter_paths = ["a", "b"], method = "breadcrumbs", gamma = 0.05
        )
        self.assertEqual(ok.gamma, 0.05)
        for bad in (-0.01, 1.0):
            with self.assertRaises(Exception):
                self.schema(
                    adapter_paths = ["a", "b"], method = "breadcrumbs", gamma = bad
                )
        # SCE select_topk must stay in (0, 1].
        ok = self.schema(
            adapter_paths = ["a", "b"], method = "sce", select_topk = 0.5
        )
        self.assertEqual(ok.select_topk, 0.5)
        for bad in (0.0, -0.1, 1.5):
            with self.assertRaises(Exception):
                self.schema(
                    adapter_paths = ["a", "b"], method = "sce", select_topk = bad
                )


if __name__ == "__main__":
    unittest.main()
