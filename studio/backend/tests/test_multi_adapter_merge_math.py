# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""PEFT delegation tests for unsloth.multi_adapter_merge.

The merge arithmetic now lives in PEFT (``add_weighted_adapter``), so these
tests pin what matters on our side: the supported method list and the mapping
from Unsloth's config onto PEFT's arguments.
"""

import importlib.util
import sys
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]


def _load_core_module():
    spec = importlib.util.spec_from_file_location(
        "multi_adapter_merge_under_test",
        _REPO_ROOT / "unsloth" / "multi_adapter_merge.py",
    )
    module = importlib.util.module_from_spec(spec)
    # Registered before exec: the module uses postponed annotations, and its
    # dataclass resolves those through sys.modules[cls.__module__].
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class TestPeftMethodMapping(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.core = _load_core_module()

    def test_supported_methods_are_peft_combination_types(self):
        self.assertEqual(
            self.core.SUPPORTED_METHODS,
            (
                "linear", "svd", "cat", "ties", "dare_ties", "dare_linear",
                "magnitude_prune",
                "ties_svd", "dare_ties_svd", "dare_linear_svd",
                "magnitude_prune_svd",
            ),
        )

    def test_density_is_forwarded_for_sparsifying_methods(self):
        for method in (
            "ties", "dare_ties", "dare_linear", "magnitude_prune",
            "ties_svd", "dare_ties_svd", "dare_linear_svd", "magnitude_prune_svd",
        ):
            kwargs = self.core._peft_combination_kwargs(method, density = 0.25)
            self.assertEqual(
                kwargs, {"combination_type": method, "density": 0.25}, method
            )

    def test_linear_and_cat_take_no_density(self):
        for method in ("linear", "cat"):
            kwargs = self.core._peft_combination_kwargs(method, density = 0.25)
            self.assertEqual(kwargs, {"combination_type": method}, method)

    def test_svd_forwards_the_target_rank(self):
        expected = {
            "svd": {"svd_rank": 32},
            # The sparsifying variants read density too, so the core's
            # default travels along with the rank.
            "ties_svd": {"density": 0.5, "svd_rank": 32},
            "dare_ties_svd": {"density": 0.5, "svd_rank": 32},
            "dare_linear_svd": {"density": 0.5, "svd_rank": 32},
            "magnitude_prune_svd": {"density": 0.5, "svd_rank": 32},
        }
        for method, extra in expected.items():
            kwargs = self.core._peft_combination_kwargs(method, target_rank = 32)
            self.assertEqual(
                kwargs,
                {"combination_type": method, **extra},
                method,
            )

    def test_svd_without_a_target_rank_uses_pefts_default(self):
        expected = {
            "svd": {},
            "ties_svd": {"density": 0.5},
            "dare_ties_svd": {"density": 0.5},
            "dare_linear_svd": {"density": 0.5},
            "magnitude_prune_svd": {"density": 0.5},
        }
        for method, extra in expected.items():
            kwargs = self.core._peft_combination_kwargs(method)
            self.assertEqual(
                kwargs, {"combination_type": method, **extra}, method
            )

    def test_svd_ignores_density(self):
        kwargs = self.core._peft_combination_kwargs("svd", density = 0.25)
        self.assertNotIn("density", kwargs)

    def test_svd_variants_take_both_knobs(self):
        for method in (
            "ties_svd", "dare_ties_svd", "dare_linear_svd", "magnitude_prune_svd",
        ):
            kwargs = self.core._peft_combination_kwargs(
                method, density = 0.25, target_rank = 32
            )
            self.assertEqual(
                kwargs,
                {"combination_type": method, "density": 0.25, "svd_rank": 32},
                method,
            )


class TestConfigAliases(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.core = _load_core_module()

    def _config(self, **kwargs):
        kwargs.setdefault("adapter_paths", ["a", "b"])
        kwargs.setdefault("weights", [1.0, 1.0])
        return self.core.MultiAdapterMergeConfig(**kwargs)

    def test_dare_alias_maps_to_dare_ties(self):
        self.assertEqual(self._config(method = "dare").method, "dare_ties")

    def test_ctm_alias_maps_to_svd(self):
        self.assertEqual(self._config(method = "ctm").method, "svd")

    def test_hyphenated_names_are_normalised(self):
        self.assertEqual(self._config(method = "DARE-TIES").method, "dare_ties")

    def test_removed_in_house_methods_are_rejected(self):
        for method in ("sce", "della", "breadcrumbs", "multislerp", "model_stock"):
            with self.assertRaises(ValueError, msg = method):
                self._config(method = method)

    def test_density_must_stay_in_range(self):
        for density in (0.0, -0.1, 1.1):
            with self.assertRaises(ValueError):
                self._config(density = density)

    def test_target_rank_must_be_positive(self):
        for target_rank in (0, -4):
            with self.assertRaises(ValueError):
                self._config(target_rank = target_rank)


if __name__ == "__main__":
    unittest.main()