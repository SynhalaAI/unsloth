# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Reference-parity checks for every merge method in unsloth.multi_adapter_merge.

Each method is compared against an independent reimplementation written from its
published reference: mergekit for linear/ties/dare/sparsify/sce/multislerp/
model_stock, PEFT for magnitude_prune/cat, and the DARE paper for the
1/(1-p) rescale. The references were transcribed from the upstream sources, not
from this repo, so agreement is evidence about the algorithm and not about two
copies of the same code.

Three deviations are pinned here on purpose, so a future change to either side
fails loudly instead of silently drifting:

* the TIES family implements mergekit's sum consensus plus weight divisor but
  not mergekit's optional L1 ``rescale``; the comparators below share that
  choice, and ``test_ties_without_the_l1_rescale_is_what_the_core_implements``
  measures the difference it makes;
* the magnitude trims keep every element tied at the threshold, where mergekit
  keeps exactly k;
* multislerp falls back to a linear mean when the deltas are antipodal, where
  mergekit raises for more than two inputs.
"""

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

_REPO_ROOT = Path(__file__).resolve().parents[3]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    # Registered before exec: the module uses postponed annotations, and
    # dataclasses resolves those through sys.modules[cls.__module__].
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope = "module")
def core():
    return _load(
        "multi_adapter_merge_reference_parity",
        _REPO_ROOT / "unsloth" / "multi_adapter_merge.py",
    )


# ---------------------------------------------------------------------------
# Reference implementations (transcribed from the upstream sources)
# ---------------------------------------------------------------------------


def ref_linear(deltas, weights):
    """mergekit linear: sum(w_i * tensor_i), with weights pre-normalised to 1."""
    return sum(delta * weight for delta, weight in zip(deltas, weights))


def ref_magnitude(tensor, density):
    """mergekit sparsify.magnitude with rescale_norm=None (exactly k kept)."""
    if density >= 1:
        return tensor
    k = int(density * tensor.numel())
    mask = torch.zeros(tensor.numel(), dtype=tensor.dtype)
    flat = tensor.abs().reshape(-1)
    topk = torch.argsort(flat, descending=True)[:k]
    mask[topk] = 1
    return tensor * mask.reshape_as(tensor)


def ref_bernoulli(tensor, keep, generator):
    """mergekit sparsify.bernoulli (dare) without the optional rescale."""
    mask = torch.bernoulli(
        torch.full_like(tensor, fill_value=keep, dtype=torch.float32), generator=generator
    )
    return tensor.float() * mask


def ref_sign_mask(weighted):
    """mergekit get_mask(method="sum"): sign == majority sign, 0 votes positive."""
    sign = weighted.sign()
    majority = (weighted.sum(dim=0) >= 0).to(weighted.dtype) * 2 - 1
    return sign == majority


def ref_ties(deltas, weights, density, trim=ref_magnitude):
    """mergekit TIES: trim, weight, sum-consensus mask, weight divisor."""
    trimmed = [trim(delta, density) for delta in deltas]
    weighted = torch.stack([t.float() * w for t, w in zip(trimmed, weights)])
    mask = ref_sign_mask(weighted)
    mixed = (weighted * mask).sum(dim=0)
    # mergekit: divisor = (weights * mask).sum(dim=0), zeros -> 1. The mask is
    # identical across adapters, so the weight axis has to be broadcast.
    weight_column = torch.tensor(weights, dtype=torch.float32).view(
        -1, *([1] * (mask.dim() - 1))
    )
    divisor = (weight_column * mask.to(torch.float32)).sum(dim=0)
    divisor = torch.where(divisor == 0, torch.ones_like(divisor), divisor)
    return mixed / divisor

def ref_della_sparsify(tensor, density, epsilon, generator):
    """mergekit sparsify.della_magprune without the optional rescale."""
    if density >= 1:
        return tensor
    original_shape = tensor.shape
    work = tensor.to(torch.float32)
    if work.dim() < 2:
        work = work.unsqueeze(0)
    magnitudes = work.abs()
    sorted_indices = torch.argsort(magnitudes, dim=1, descending=False)
    ranks = sorted_indices.argsort(dim=1).to(torch.float32) + 1
    min_ranks = ranks.min(dim=1, keepdim=True).values
    max_ranks = ranks.max(dim=1, keepdim=True).values
    rank_norm = ((ranks - min_ranks) / (max_ranks - min_ranks)).clamp(0, 1)
    probs = (density - epsilon) + rank_norm * 2 * epsilon
    mask = torch.bernoulli(probs, generator=generator)
    return (work * mask).reshape(original_shape)


def ref_magnitude_outliers(tensor, density, gamma):
    """mergekit sparsify.magnitude_outliers (breadcrumbs), no rescale."""
    if density >= 1:
        return tensor
    num_elems = tensor.numel()
    target_n = int(density * num_elems)
    n_top = int(gamma * num_elems)
    n_bot = max(0, num_elems - target_n - n_top)
    flat = tensor.abs().reshape(-1).to(torch.float32)
    indices = torch.sort(flat, descending=False).indices
    mask = torch.zeros(num_elems, dtype=torch.float32)
    mask[indices[n_bot : n_bot + target_n]] = 1
    return tensor.to(torch.float32) * mask.reshape_as(tensor)


def ref_sce(deltas, select_topk):
    """mergekit sce: variance select, energy weights, erase, weight divisor."""
    stack = torch.stack([delta.to(torch.float32) for delta in deltas], dim=0)
    if select_topk < 1:
        var = torch.var(stack, dim=0, unbiased=False)
        nonzero = torch.count_nonzero(var)
        k = int(nonzero.item() * select_topk)
        if k == 0:
            return torch.zeros_like(stack[0])
        _, indices = torch.topk(var.abs().view(-1), k=k, largest=True)
        selection = torch.zeros_like(var)
        selection.view(-1)[indices] = 1
        stack = stack * selection.unsqueeze(0)
    energies = stack.square().reshape(stack.shape[0], -1).mean(dim=1)
    total = energies.sum()
    if abs(total.item()) < 1e-6:
        tv_weights = torch.ones_like(energies) / energies.shape[0]
    else:
        tv_weights = energies / total
    erase = ref_sign_mask(stack).to(torch.float32)
    while tv_weights.dim() < stack.dim():
        tv_weights = tv_weights.unsqueeze(-1)
    erased = tv_weights * erase
    merged = (stack * erased).sum(dim=0)
    return merged / erased.sum(dim=0).clamp(min=1e-6)


def ref_multislerp(deltas, weights, eps=1e-8):
    """mergekit multislerp: tangent-space barycentric interpolation, base=0."""
    stack = torch.stack([delta.to(torch.float32) for delta in deltas], dim=0)
    if stack.shape[0] == 1:
        return stack[0]
    flat = stack.view(stack.shape[0], -1)
    weights = torch.tensor(weights, dtype=torch.float32)
    weights = weights / weights.sum()
    norms = torch.norm(flat, dim=-1, keepdim=True)
    unit = flat / (norms + eps)
    mean = (unit * weights.view(-1, 1)).sum(0)
    mean_norm = torch.norm(mean)
    if mean_norm < eps:
        return (flat * weights.view(-1, 1)).sum(0).view(stack.shape[1:])
    mean = mean / mean_norm
    dots = (unit * mean).sum(-1, keepdim=True)
    tangent = unit - dots * mean
    tangent_result = (tangent * weights.view(-1, 1)).sum(0)
    tangent_norm = torch.norm(tangent_result) + eps
    result = mean * torch.cos(tangent_norm) + tangent_result * (
        torch.sin(tangent_norm) / tangent_norm
    )
    avg_norm = (norms.squeeze(-1) * weights).sum()
    return (result * avg_norm).view(stack.shape[1:])


def ref_model_stock(deltas):
    """mergekit model_stock, in delta space (base = 0): t * mean(deltas)."""
    flats = [delta.reshape(-1).to(torch.float32) for delta in deltas]
    cosines = []
    for index, a in enumerate(flats):
        for b in flats[index + 1 :]:
            product = torch.norm(a) * torch.norm(b)
            cosines.append(((a * b).sum() / product.clamp(min=1e-6)).clamp(-1, 1))
    cos_theta = torch.stack(cosines).mean(dim=0)
    count = len(flats)
    denominator = 1 + (count - 1) * cos_theta
    if denominator.abs() < 1e-6:
        return torch.zeros_like(deltas[0])
    t = (count * cos_theta) / denominator
    return (t * (sum(flats) / count)).reshape(deltas[0].shape)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_DELTAS = [
    torch.tensor([[1.0, -2.0, 3.0, -0.5], [0.25, 4.0, -1.0, 2.0]]),
    torch.tensor([[2.0, 1.0, -3.0, 0.75], [-0.5, 3.0, 2.0, -4.0]]),
    torch.tensor([[0.5, 2.0, 1.0, -1.5], [1.5, -2.5, 3.5, 1.0]]),
]
_WEIGHTS = [0.5, 0.3, 0.2]


class TestLinear:
    """The dict-level ``_linear_merge`` (the streaming path mirrors this)."""

    def test_matches_the_weighted_sum(self, core):
        as_dicts = [{"layer.weight": delta} for delta in _DELTAS]
        merged = core._linear_merge(as_dicts, list(_WEIGHTS))
        expected = ref_linear(_DELTAS, _WEIGHTS)
        assert torch.allclose(merged["layer.weight"], expected, atol = 1e-6)

    def test_missing_modules_count_as_a_zero_delta(self, core):
        first = {"a": torch.ones(2, 2), "b": torch.ones(2, 2)}
        second = {"a": torch.full((2, 2), 3.0)}
        merged = core._linear_merge([first, second], [0.25, 0.75])
        assert torch.allclose(merged["a"], 0.25 * first["a"] + 0.75 * second["a"])
        assert torch.allclose(merged["b"], 0.25 * first["b"])

    def test_one_adapter_is_scaled_by_its_weight(self, core):
        merged = core._linear_merge([{"a": torch.ones(2, 2)}], [0.4])
        assert torch.allclose(merged["a"], torch.full((2, 2), 0.4))


class TestTies:
    @staticmethod
    def _tie_free(count = 3, rows = 6, cols = 5, seed = 101):
        """Deltas with no repeated magnitude.

        The core's trim keeps every element equal to the k-th magnitude while
        mergekit keeps exactly k, so the two can only be compared on data with
        distinct magnitudes. The tie behaviour itself is pinned separately by
        test_trim_keeps_every_element_tied_at_the_threshold.
        """
        generator = torch.Generator().manual_seed(seed)
        return [torch.randn(rows, cols, generator = generator) for _ in range(count)]

    def test_matches_mergekit_trim_elect_and_weight_divisor(self, core):
        for density in (1.0, 0.7, 0.5, 0.3):
            deltas = self._tie_free(seed = 101 + int(density * 100))
            merged = core._ties_merge_key(deltas, list(_WEIGHTS), density = density)
            expected = ref_ties(deltas, _WEIGHTS, density)
            assert torch.allclose(merged, expected, atol = 1e-6), density

    def test_density_one_is_the_sign_voted_weighted_mean(self, core):
        deltas = self._tie_free(seed = 202)
        merged = core._ties_merge_key(deltas, list(_WEIGHTS), density = 1.0)
        assert torch.allclose(merged, ref_ties(deltas, _WEIGHTS, 1.0), atol = 1e-6)

    def test_aligned_adapters_average_to_the_weighted_sum(self, core):
        # All-positive deltas: every trimmed value survives the vote, so TIES
        # reduces to the weighted sum (divisor = sum of weights = 1).
        positive = [delta.abs() for delta in _DELTAS]
        merged = core._ties_merge_key(positive, list(_WEIGHTS), density = 1.0)
        assert torch.allclose(merged, ref_linear(positive, _WEIGHTS), atol = 1e-5)

    def test_the_vote_excludes_zeroed_entries(self, core):
        # A coordinate that is zero after trimming must not vote and must not
        # contribute: its weight is excluded from the divisor too. Magnitudes
        # are distinct so the trim is the same on both sides.
        first = torch.tensor([1.0, -1.0, 0.0, 0.0])
        second = torch.tensor([1.0, 1.0, 5.0, 0.0])
        merged = core._ties_merge_key([first, second], [0.5, 0.5], density = 1.0)
        # density 1 keeps every non-zero value; the elected sign at position 1 is
        # a zero-sum, which the core resolves to +1, and only `second` agrees.
        assert merged[0] == pytest.approx(1.0)
        assert merged[1] == pytest.approx(1.0)
        assert merged[2] == pytest.approx(5.0)
        assert merged[3] == pytest.approx(0.0)
        assert torch.allclose(merged, ref_ties([first, second], [0.5, 0.5], 1.0))

    def test_never_returns_nan_when_nothing_aligns(self, core):
        first = torch.tensor([1.0, 1.0])
        second = torch.tensor([-1.0, -1.0])
        merged = core._ties_merge_key([first, second], [0.5, 0.5], density = 1.0)
        assert torch.isfinite(merged).all()

    def test_trim_keeps_every_element_tied_at_the_threshold(self, core):
        # Documented deviation: mergekit keeps exactly k, the core keeps k plus
        # every element equal to the k-th magnitude. With all magnitudes equal
        # and density 0.25, mergekit keeps one element per adapter and the core
        # keeps all four, so the two disagree on three coordinates.
        tied = torch.tensor([1.0, 1.0, 1.0, 1.0])
        other = torch.tensor([2.0, 2.0, 2.0, 2.0])
        merged = core._ties_merge_key([tied, other], [0.5, 0.5], density = 0.25)
        assert torch.allclose(merged, torch.full((4,), 1.5))
        reference = ref_ties([tied, other], [0.5, 0.5], 0.25)
        assert not torch.allclose(merged, reference, atol = 1e-6)
        assert int((reference != merged).sum().item()) == 3

    def test_without_the_l1_rescale_is_what_the_core_implements(self, core):
        # The core implements mergekit's consensus + divisor, not mergekit's
        # optional RescaleNorm.l1. Pin how far apart those are, so enabling the
        # rescale upstream is a deliberate change here rather than a surprise.
        deltas = self._tie_free(rows = 32, cols = 32, seed = 303)
        merged = core._ties_merge_key(deltas, list(_WEIGHTS), density = 0.5)

        def rescale_l1(tensor, density):
            masked = ref_magnitude(tensor, density)
            before = tensor.abs().sum()
            after = masked.abs().sum()
            return masked if float(after) == 0.0 else masked * (before / after)

        rescaled = ref_ties(deltas, _WEIGHTS, 0.5, trim = rescale_l1)
        assert torch.allclose(merged, ref_ties(deltas, _WEIGHTS, 0.5), atol = 1e-6)
        relative_gap = float(
            (merged - rescaled).norm() / rescaled.norm()
        )
        assert 0.01 < relative_gap < 1.0

class TestDare:
    """DARE: drop at rate p, rescale by 1/(1-p), then the linear / TIES tail."""

    SEED = 7

    @staticmethod
    def _keep_masks(deltas, drop_rate, seed):
        # The core seeds one generator per call and draws it in adapter order,
        # so the reference draws from an identically seeded generator to make
        # the comparison exact rather than statistical.
        generator = torch.Generator().manual_seed(seed)
        return [
            torch.bernoulli(
                torch.full(delta.shape, 1.0 - drop_rate, dtype = torch.float32),
                generator = generator,
            )
            for delta in deltas
        ]

    def test_dare_linear_matches_drop_rescale_then_weighted_sum(self, core):
        masks = self._keep_masks(_DELTAS, 0.4, self.SEED)
        scale = 1.0 / (1.0 - 0.4)
        expected = sum(
            delta * mask * scale * weight
            for delta, mask, weight in zip(_DELTAS, masks, _WEIGHTS)
        )
        merged = core._dare_linear_merge_key(
            list(_DELTAS), list(_WEIGHTS), drop_rate = 0.4, seed = self.SEED
        )
        assert torch.allclose(merged, expected, atol = 1e-6)

    def test_dare_ties_matches_drop_rescale_then_ties(self, core):
        masks = self._keep_masks(_DELTAS, 0.4, self.SEED)
        scale = 1.0 / (1.0 - 0.4)
        rescaled = [
            delta * mask * scale for delta, mask in zip(_DELTAS, masks)
        ]
        expected = ref_ties(rescaled, _WEIGHTS, 0.5)
        merged = core._dare_ties_merge_key(
            list(_DELTAS), list(_WEIGHTS), density = 0.5, drop_rate = 0.4, seed = self.SEED
        )
        assert torch.allclose(merged, expected, atol = 1e-6)

    def test_zero_drop_rate_is_the_plain_merge(self, core):
        dare = core._dare_linear_merge_key(
            list(_DELTAS), list(_WEIGHTS), drop_rate = 0.0, seed = self.SEED
        )
        assert torch.allclose(dare, ref_linear(_DELTAS, _WEIGHTS), atol = 1e-6)
        ties = core._dare_ties_merge_key(
            list(_DELTAS), list(_WEIGHTS), density = 1.0, drop_rate = 0.0, seed = self.SEED
        )
        assert torch.allclose(ties, ref_ties(_DELTAS, _WEIGHTS, 1.0), atol = 1e-6)

    def test_the_masks_are_independent_per_adapter(self, core):
        # One generator per adapter, drawn in order: a shared mask would leave
        # the same weights dropped in every adapter.
        masks = self._keep_masks(_DELTAS, 0.3, self.SEED)
        assert not torch.equal(masks[0], masks[1])
        merged = core._dare_linear_merge_key(
            list(_DELTAS), list(_WEIGHTS), drop_rate = 0.3, seed = self.SEED
        )
        expected = sum(
            delta * mask * (1.0 / 0.7) * weight
            for delta, mask, weight in zip(_DELTAS, masks, _WEIGHTS)
        )
        assert torch.allclose(merged, expected, atol = 1e-6)


class TestMagnitudePrune:
    def test_matches_pefts_topk_prune_then_weighted_sum(self, core):
        # Continuous values, so no two magnitudes tie: with a tie the core keeps
        # more than k elements by design (see the TIES tie test).
        generator = torch.Generator().manual_seed(17)
        deltas = [torch.randn(5, 6, generator = generator) for _ in range(3)]
        expected = sum(
            ref_magnitude(delta, 0.5) * weight
            for delta, weight in zip(deltas, _WEIGHTS)
        )
        merged = core._magnitude_prune_merge_key(
            list(deltas), list(_WEIGHTS), density = 0.5
        )
        assert torch.allclose(merged, expected, atol = 1e-6)

    def test_full_density_keeps_everything(self, core):
        merged = core._magnitude_prune_merge_key(
            list(_DELTAS), list(_WEIGHTS), density = 1.0
        )
        assert torch.allclose(merged, ref_linear(_DELTAS, _WEIGHTS), atol = 1e-6)

    def test_the_pruned_share_is_the_requested_density(self, core):
        delta = torch.randn(40, 40, generator = torch.Generator().manual_seed(3))
        merged = core._magnitude_prune_merge_key([delta, delta], [0.5, 0.5], density = 0.25)
        nonzero = int((merged != 0).sum().item())
        # Two identical adapters, both pruned to the same 25%: 0.25 * 1600.
        assert nonzero == int(0.25 * delta.numel())

class TestCat:
    def test_factor_concatenation_equals_the_weighted_sum(self, core):
        # PEFT cat: concat(sqrt(w*s) B) @ concat(sqrt(w*s) A) expands to the
        # same weighted sum in weight space.
        generator = torch.Generator().manual_seed(11)
        factors = []
        for rank, scaling in ((2, 1.0), (3, 2.0)):
            A = torch.randn(rank, 5, generator = generator)
            B = torch.randn(4, rank, generator = generator)
            factors.append((A, B, scaling, 0.6 if rank == 2 else 0.4))
        merged = core._cat_merge_key(list(factors))
        expected = sum(
            weight * scaling * (B @ A) for A, B, scaling, weight in factors
        )
        assert torch.allclose(merged, expected, atol = 1e-5)

    def test_a_negative_weight_flips_one_factor_only(self, core):
        # B @ A is 2 per element (rank 2, all ones). The negative weight scales
        # its own contribution by -0.5, the positive one by 1.5.
        A = torch.ones(2, 3)
        B = torch.ones(3, 2)
        merged = core._cat_merge_key([(A, B, 1.0, -0.5), (A, B, 1.0, 1.5)])
        assert torch.allclose(merged, torch.full((3, 3), 2.0))

    def test_heterogeneous_ranks_concatenate(self, core):
        generator = torch.Generator().manual_seed(13)
        first = (torch.randn(1, 4, generator = generator), torch.randn(6, 1, generator = generator), 1.0, 1.0)
        second = (torch.randn(4, 4, generator = generator), torch.randn(6, 4, generator = generator), 1.0, 0.0)
        merged = core._cat_merge_key([first, second])
        assert torch.allclose(merged, first[3] * first[2] * (first[1] @ first[0]), atol = 1e-5)


class TestCtm:
    def test_without_a_rank_it_is_the_weighted_sum(self, core):
        merged = core._ctm_merge_key(list(_DELTAS), list(_WEIGHTS), target_rank = None)
        assert torch.allclose(merged, ref_linear(_DELTAS, _WEIGHTS), atol = 1e-6)

    def test_a_rank_projects_onto_the_best_rank_r_approximation(self, core):
        merged = core._ctm_merge_key(list(_DELTAS), list(_WEIGHTS), target_rank = 1)
        target = ref_linear(_DELTAS, _WEIGHTS)
        U, S, Vh = torch.linalg.svd(target.to(torch.float32), full_matrices = False)
        best_rank_1 = (U[:, :1] * S[:1]) @ Vh[:1, :]
        assert torch.allclose(merged, best_rank_1, atol = 1e-5)
        # Rank really is 1, and the truncation loses exactly the tail energy.
        assert int(torch.linalg.matrix_rank(merged).item()) == 1
        tail = (S[1:] ** 2).sum()
        assert torch.allclose(
            torch.tensor(float(((merged - target) ** 2).sum())), tail, rtol = 1e-4
        )

    def test_a_rank_at_or_above_the_shape_keeps_the_sum(self, core):
        merged = core._ctm_merge_key(list(_DELTAS), list(_WEIGHTS), target_rank = 99)
        assert torch.allclose(merged, ref_linear(_DELTAS, _WEIGHTS), atol = 1e-5)

    def test_a_rank_below_one_dimension_is_ignored(self, core):
        # Only 2-D deltas can be compressed; a 1-D delta is returned as-is.
        vector = torch.tensor([1.0, -2.0, 3.0])
        merged = core._ctm_merge_key([vector], [1.0], target_rank = 1)
        assert torch.allclose(merged, vector)

class TestSce:
    def test_matches_mergekit_select_consensus_erase(self, core):
        merged = core._sce_merge_key(list(_DELTAS), list(_WEIGHTS), select_topk = 0.5)
        assert torch.allclose(merged, ref_sce(_DELTAS, 0.5), atol = 1e-6)

    def test_user_weights_are_ignored_by_design(self, core):
        # SCE derives weights from delta energy, which is why the API marks it
        # as an auto-weight method.
        first = core._sce_merge_key(list(_DELTAS), [0.9, 0.05, 0.05], select_topk = 1.0)
        second = core._sce_merge_key(list(_DELTAS), [0.1, 0.8, 0.1], select_topk = 1.0)
        assert torch.allclose(first, second)

    def test_select_topk_one_keeps_every_element(self, core):
        merged = core._sce_merge_key(list(_DELTAS), list(_WEIGHTS), select_topk = 1.0)
        assert torch.allclose(merged, ref_sce(_DELTAS, 1.0), atol = 1e-6)

    def test_identical_adapters_merge_to_themselves(self, core):
        delta = _DELTAS[0]
        merged = core._sce_merge_key([delta, delta], [0.5, 0.5], select_topk = 1.0)
        assert torch.allclose(merged, delta, atol = 1e-6)


class TestDella:
    SEED = 5

    @staticmethod
    def _pruned(deltas, density, epsilon, seed):
        generator = torch.Generator().manual_seed(seed)
        return [
            ref_della_sparsify(delta, density, epsilon, generator) for delta in deltas
        ]

    def test_matches_mergekit_della_magprune_then_ties(self, core):
        pruned = self._pruned(_DELTAS, 0.5, 0.15, self.SEED)
        expected = ref_ties(pruned, _WEIGHTS, 1.0)
        merged = core._della_merge_key(
            list(_DELTAS), list(_WEIGHTS),
            density = 0.5, della_epsilon = 0.15, seed = self.SEED, sign_elect = True,
        )
        assert torch.allclose(merged, expected, atol = 1e-6)

    def test_della_linear_matches_the_prune_then_weighted_sum(self, core):
        pruned = self._pruned(_DELTAS, 0.5, 0.15, self.SEED)
        expected = ref_linear(pruned, _WEIGHTS)
        merged = core._della_linear_merge_key(
            list(_DELTAS), list(_WEIGHTS),
            density = 0.5, della_epsilon = 0.15, seed = self.SEED,
        )
        assert torch.allclose(merged, expected, atol = 1e-6)

    def test_the_keep_probability_spans_density_plus_minus_epsilon(self, core):
        # Rank 1 in a row gets density - epsilon, the top rank density + epsilon.
        row = torch.tensor([[1.0, 2.0, 3.0, 4.0, 100.0]])
        generator = torch.Generator().manual_seed(0)
        keep_share = []
        for _ in range(300):
            keep_share.append(
                float(
                    ref_della_sparsify(row, 0.5, 0.2, generator)
                    .abs()
                    .gt(0)
                    .float()
                    .mean()
                )
            )
        mean_keep = sum(keep_share) / len(keep_share)
        assert 0.5 - 0.05 < mean_keep < 0.5 + 0.05


class TestBreadcrumbs:
    def test_matches_mergekit_magnitude_outliers(self, core):
        expected = sum(
            ref_magnitude_outliers(delta, 0.5, 0.01) * weight
            for delta, weight in zip(_DELTAS, _WEIGHTS)
        )
        merged = core._breadcrumbs_merge_key(
            list(_DELTAS), list(_WEIGHTS), density = 0.5, gamma = 0.01, sign_elect = False
        )
        assert torch.allclose(merged, expected, atol = 1e-6)

    def test_ties_variant_adds_the_sign_election(self, core):
        pruned = [ref_magnitude_outliers(delta, 0.5, 0.01) for delta in _DELTAS]
        expected = ref_ties(pruned, _WEIGHTS, 1.0)
        merged = core._breadcrumbs_ties_merge_key(
            list(_DELTAS), list(_WEIGHTS), density = 0.5, gamma = 0.01
        )
        assert torch.allclose(merged, expected, atol = 1e-6)

    def test_gamma_removes_the_outliers_first(self, core):
        # One huge outlier in 100 elements: with gamma=0.05 it is dropped, and
        # the kept mass is the density share below it. Two adapters, because a
        # single adapter short-circuits to a plain scale (see the shared-contract
        # single-adapter test).
        delta = torch.ones(1, 100)
        delta[0, 0] = 1000.0
        merged = core._breadcrumbs_merge_key(
            [delta, delta.clone()], [0.5, 0.5], density = 0.5, gamma = 0.05,
            sign_elect = False,
        )
        assert float(merged.abs().max()) == 1.0
        assert int((merged != 0).sum().item()) == 50

class TestMultislerp:
    def test_matches_mergekit_tangent_space_interpolation(self, core):
        merged = core._multislerp_merge_key(list(_DELTAS), list(_WEIGHTS))
        assert torch.allclose(merged, ref_multislerp(_DELTAS, _WEIGHTS), atol = 1e-6)

    def test_parallel_deltas_average_to_the_weighted_mean(self, core):
        # Same direction: the sphere interpolation reduces to the mean of norms
        # along that direction.
        base = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        merged = core._multislerp_merge_key([base, base * 2.0], [0.5, 0.5])
        assert torch.allclose(merged, ref_multislerp([base, base * 2.0], [0.5, 0.5]), atol = 1e-6)
        assert torch.allclose(merged, base * 1.5, atol = 1e-5)

    def test_antipodal_deltas_fall_back_to_a_linear_mean(self, core):
        # Documented deviation: mergekit raises for >2 antipodal inputs, the
        # core returns their (zero) linear mean.
        first = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        second = -first
        merged = core._multislerp_merge_key([first, second], [0.5, 0.5])
        assert merged.shape == first.shape
        assert torch.allclose(merged, torch.zeros_like(first))

    def test_the_antipodal_fallback_keeps_a_one_dimensional_shape(self, core):
        # A one-dimensional delta (a bias-shaped delta, not produced by B @ A of
        # a LoRA Linear but accepted by this function) comes back with an extra
        # leading axis: the fallback does tensors * weights.view(-1, 1, 1).
        # mergekit reshapes with .view(tensors.shape[1:]) before returning.
        first = torch.tensor([1.0, 0.0])
        merged = core._multislerp_merge_key([first, -first], [0.5, 0.5])
        assert merged.shape == first.shape

    def test_the_norm_is_the_weighted_average_of_input_norms(self, core):
        first = torch.tensor([3.0, 0.0])
        second = torch.tensor([0.0, 5.0])
        merged = core._multislerp_merge_key([first, second], [0.5, 0.5])
        assert float(torch.linalg.norm(merged)) == pytest.approx(4.0, rel = 1e-5)


class TestModelStock:
    def test_matches_the_model_stock_formula(self, core):
        merged = core._model_stock_merge_key(list(_DELTAS), list(_WEIGHTS))
        assert torch.allclose(merged, ref_model_stock(_DELTAS), atol = 1e-6)

    def test_an_aligned_stock_keeps_the_full_average(self, core):
        # cos(theta) -> 1 makes t -> 1, so the answer is the plain mean.
        base = torch.tensor([[1.0, -2.0], [3.0, 0.5]])
        deltas = [base, base * 1.0, base * 1.0]
        merged = core._model_stock_merge_key(deltas, [1.0, 1.0, 1.0])
        assert torch.allclose(merged, base, atol = 1e-4)

    def test_orthogonal_deltas_collapse_toward_the_base(self, core):
        # Pairwise cos(theta) = 0 gives t = 0: no shared direction, so nothing
        # is added. Three genuinely orthogonal axes, not an antipodal pair.
        deltas = [
            torch.tensor([1.0, 0.0, 0.0]),
            torch.tensor([0.0, 1.0, 0.0]),
            torch.tensor([0.0, 0.0, 1.0]),
        ]
        merged = core._model_stock_merge_key(deltas, [1.0, 1.0, 1.0])
        assert torch.allclose(merged, torch.zeros(3), atol = 1e-6)

    def test_fewer_than_three_adapters_is_rejected(self, core):
        with pytest.raises(ValueError):
            core._model_stock_merge_key(list(_DELTAS[:2]), [0.5, 0.5])

    def test_balanced_opposed_deltas_do_not_explode(self, core):
        # denominator -> 0: the core keeps the base instead of dividing by ~0.
        deltas = [
            torch.tensor([1.0, 0.0]),
            torch.tensor([-1.0, 0.0]),
            torch.tensor([1.0, 0.0]),
        ]
        merged = core._model_stock_merge_key(deltas, [1.0, 1.0, 1.0])
        assert torch.isfinite(merged).all()


class TestSharedContract:
    """Properties every per-module merge key has to satisfy."""

    METHODS = [
        ("ties", lambda core, d, w: core._ties_merge_key(d, w, density = 0.5)),
        ("dare_ties", lambda core, d, w: core._dare_ties_merge_key(d, w, density = 0.5, drop_rate = 0.3, seed = 1)),
        ("dare_linear", lambda core, d, w: core._dare_linear_merge_key(d, w, drop_rate = 0.3, seed = 1)),
        ("magnitude_prune", lambda core, d, w: core._magnitude_prune_merge_key(d, w, density = 0.5)),
        # target_rank=None so the shared contract tests the merge, not the SVD;
        # TestCtm covers the compression.
        ("ctm", lambda core, d, w: core._ctm_merge_key(d, w, target_rank = None)),
        ("sce", lambda core, d, w: core._sce_merge_key(d, w, select_topk = 0.5)),
        ("della", lambda core, d, w: core._della_merge_key(d, w, density = 0.5, della_epsilon = 0.15, seed = 1)),
        ("della_linear", lambda core, d, w: core._della_linear_merge_key(d, w, density = 0.5, della_epsilon = 0.15, seed = 1)),
        ("breadcrumbs", lambda core, d, w: core._breadcrumbs_merge_key(d, w, density = 0.5, gamma = 0.01)),
        ("breadcrumbs_ties", lambda core, d, w: core._breadcrumbs_ties_merge_key(d, w, density = 0.5, gamma = 0.01)),
        ("multislerp", lambda core, d, w: core._multislerp_merge_key(d, w)),
        ("model_stock", lambda core, d, w: core._model_stock_merge_key(d, w)),
    ]

    @pytest.mark.parametrize("name,merge", METHODS, ids = [m[0] for m in METHODS])
    def test_preserves_shape_and_stays_finite(self, core, name, merge):
        merged = merge(core, list(_DELTAS), list(_WEIGHTS))
        assert merged.shape == _DELTAS[0].shape
        assert torch.isfinite(merged).all(), name

    @pytest.mark.parametrize(
        "name,merge",
        [entry for entry in METHODS if entry[0] != "model_stock"],
        ids = [m[0] for m in METHODS if m[0] != "model_stock"],
    )
    def test_a_single_adapter_is_scaled_by_its_weight(self, core, name, merge):
        # Every method short-circuits one adapter to a plain scale, which also
        # means a single adapter is never sparsified. model_stock is excluded:
        # it requires three by definition.
        merged = merge(core, [_DELTAS[0]], [0.5])
        assert torch.allclose(merged, _DELTAS[0] * 0.5, atol = 1e-6), name

    @pytest.mark.parametrize("name,merge", METHODS, ids = [m[0] for m in METHODS])
    def test_heterogeneous_ranks_and_shapes_are_tolerated(self, core, name, merge):
        wide = torch.randn(3, 9, generator = torch.Generator().manual_seed(2))
        narrow = torch.randn(3, 9, generator = torch.Generator().manual_seed(4))
        merged = merge(core, [wide, narrow, narrow.clone()], [0.4, 0.3, 0.3])
        assert merged.shape == wide.shape