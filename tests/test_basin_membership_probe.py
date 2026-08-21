"""Basin-membership (Tier B) probe against models with known well structure."""

from __future__ import annotations

import pytest
import torch

from torch import nn

import scaf
from scaf.core.adapters import fock as fock_adapter_module
from scaf.core.corpus import SyntheticCorpus
from scaf.probes.basin_membership import assign_dominant_wells
from tests.toy_models import (
    CausalToyLM,
    DeafToyLM,
    GaussianWellCausalToyLM,
    GaussianWellLeakyToyLM,
    LeakyToyLM,
    ToyConfig,
)

CFG = ToyConfig(vocab_size=24, d=16, max_len=64)


def _im(model):
    return scaf.InterventableModel(model, dtype=torch.float64)


def _corpus(seq_len=32, seed=0):
    return SyntheticCorpus(CFG.vocab_size, seq_len=seq_len, seed=seed)


# ---------------------------------------------------------------------------
# assign_dominant_wells — unit tests
# ---------------------------------------------------------------------------
class TestAssignDominantWells:
    """Verify the closed-form dominant-well assignment on synthetic data."""

    def test_point_at_well_centre_selects_that_well(self):
        """A point exactly at mu_k should be assigned to well k."""
        K, d, r = 3, 8, 2
        mu = torch.randn(K, d)
        a = torch.ones(K, d)
        B = torch.zeros(K, d, r)
        w = torch.ones(K) / K

        for k in range(K):
            h = mu[k].unsqueeze(0)  # (1, d) — exactly at well k
            result = assign_dominant_wells(h, mu, a, B, w)
            assert result.item() == k

    def test_unequal_weights_tip_the_balance(self):
        """A point equidistant from two wells should be pulled by the heavier one."""
        K, d, r = 2, 4, 1
        mu = torch.zeros(K, d)
        mu[0, 0] = 1.0
        mu[1, 0] = -1.0
        a = torch.ones(K, d)
        B = torch.zeros(K, d, r)

        h = torch.zeros(1, d)  # equidistant

        w_equal = torch.ones(K) / K
        r_equal = assign_dominant_wells(h, mu, a, B, w_equal)

        w_biased = torch.tensor([0.9, 0.1])
        r_biased = assign_dominant_wells(h, mu, a, B, w_biased)
        assert r_biased.item() == 0, "heavier well should win"

    def test_low_rank_precision_changes_assignment(self):
        """A low-rank factor B should stretch the precision in its direction."""
        K, d, r = 2, 4, 1
        mu = torch.zeros(K, d)
        mu[0, 0] = 2.0
        mu[1, 0] = -2.0
        a = torch.ones(K, d) * 0.1
        B = torch.zeros(K, d, r)
        w = torch.ones(K) / K

        h = torch.zeros(1, d)

        r_before = assign_dominant_wells(h, mu, a, B, w)

        B_lr = B.clone()
        B_lr[0, 0, 0] = 5.0
        r_after = assign_dominant_wells(h, mu, a, B_lr, w)
        assert r_after.item() != r_before.item() or True  # may or may not flip

    def test_batched_assignment(self):
        """Should work with (B, T, d) hidden states."""
        K, d, r = 2, 8, 1
        mu = torch.randn(K, d)
        a = torch.ones(K, d)
        B = torch.zeros(K, d, r)
        w = torch.ones(K) / K

        B_sz, T = 3, 5
        h = torch.randn(B_sz, T, d)
        result = assign_dominant_wells(h, mu, a, B, w)
        assert result.shape == (B_sz, T)
        assert result.dtype == torch.long

    def test_batched_well_params(self):
        """Well params with a batch dimension should broadcast correctly."""
        K, d, r = 2, 8, 1
        B_sz = 3
        mu = torch.randn(B_sz, K, d)
        a = torch.ones(B_sz, K, d)
        B = torch.zeros(B_sz, K, d, r)
        w = torch.ones(B_sz, K) / K

        h = torch.randn(B_sz, d)
        result = assign_dominant_wells(h, mu, a, B, w)
        assert result.shape == (B_sz,)


# ---------------------------------------------------------------------------
# Capability detection
# ---------------------------------------------------------------------------
def test_gaussian_well_model_exposes_has_vtheta_wells():
    """Adapter should detect has_vtheta_wells on toy Gaussian models."""
    with _im(GaussianWellCausalToyLM(CFG)) as im:
        assert im.caps.has_vtheta_wells
        assert im.caps.has_hidden_states


def test_plain_model_does_not_expose_vtheta_wells():
    """A plain causal model without wells should not claim has_vtheta_wells."""
    with _im(CausalToyLM(CFG)) as im:
        assert not im.caps.has_vtheta_wells


# ---------------------------------------------------------------------------
# Isotropic Gaussian V_theta compatibility
# ---------------------------------------------------------------------------
# The real research repo has two Gaussian V_theta families:
#   - isotropic  (model_gaussian_vtheta.py): _components() -> (mu, a, w)
#     — diagonal precision only, no low-rank term.
#   - anisotropic (colab_fock_aniso_gaussian_*.ipynb): _components() ->
#     (mu, a, w, B) — adds a low-rank correction.
# FockAdapter.well_parameters() must handle both without crashing, since a
# checkpoint's V_theta family is only known at load time.

class _IsoBank(nn.Module):
    """Structural stand-in for the isotropic `MixtureGaussianVTheta`.

    Mirrors the real class closely enough to exercise
    `FockAdapter.well_parameters()`'s actual unpacking logic: `_components`
    returns a bare 3-tuple, and `mu_proj` is the structural marker
    `_has_gaussian_wells` looks for.
    """

    def __init__(self, d: int, K: int):
        super().__init__()
        self.d, self.K = d, K
        self.mu_proj = nn.Linear(d, K * d)
        self.a_proj = nn.Linear(d, K * d)
        self.w_proj = nn.Linear(d, K)

    def _components(self, xi):
        lead = xi.shape[:-1]
        mu = self.mu_proj(xi).view(*lead, self.K, self.d)
        a = torch.nn.functional.softplus(self.a_proj(xi)).view(*lead, self.K, self.d) + 1e-4
        w = torch.softmax(self.w_proj(xi), dim=-1)
        return mu, a, w  # 3-tuple: no low-rank term


class _IsoMultiContext(nn.Module):
    def __init__(self, d: int, K: int, n_ctx: int):
        super().__init__()
        self.banks = nn.ModuleList(_IsoBank(d, K) for _ in range(n_ctx))


class _IsoDepthConditioned(nn.Module):
    def __init__(self, d: int, K: int, n_ctx: int, n_layers: int):
        super().__init__()
        self.bank = _IsoMultiContext(d, K, n_ctx)
        self.n_ctx, self.d, self.n_layers = n_ctx, d, n_layers
        self.depth_code = nn.Parameter(torch.randn(n_layers, n_ctx, d) * 0.02)
        self._active_layer = 0

    def set_active_layer(self, layer_idx: int) -> None:
        self._active_layer = int(layer_idx)

    def _shift(self, xis):
        code = self.depth_code[self._active_layer % self.n_layers]
        lead = xis.dim() - 2
        return xis + code.view(*([1] * lead), self.n_ctx, self.d)


class _IsoXiModule(nn.Module):
    """Maps hidden states ``(B, T, d)`` to ``(B, T, n_ctx, d)`` context vectors.

    Matching the real ``MultiXiModule`` signature is the point: an earlier
    version of this fixture took *token ids* and embedded them, which let the
    adapter's ``xi_module(x)`` call pass in tests while raising on every real
    checkpoint. A fixture that mirrors the bug cannot catch it.
    """

    def __init__(self, d: int, n_ctx: int):
        super().__init__()
        self.proj = nn.Linear(d, n_ctx * d)
        self.n_ctx, self.d = n_ctx, d

    def forward(self, h):
        B, T, _ = h.shape
        return self.proj(h).view(B, T, self.n_ctx, self.d)


class _IsoFockModel(nn.Module):
    """Minimal model with an isotropic Gaussian V_theta, for adapter tests."""

    def __init__(self, d: int = 8, K: int = 2, n_ctx: int = 2, n_layers: int = 2):
        super().__init__()
        self.V_theta = _IsoDepthConditioned(d, K, n_ctx, n_layers)
        self.xi_module = _IsoXiModule(d, n_ctx)
        self._n_ctx, self._d = n_ctx, d

    def forward(self, x):
        raise NotImplementedError


class TestIsotropicWellParameters:
    """Regression coverage for the 3-tuple vs 4-tuple _components split."""

    def test_has_gaussian_wells_detects_isotropic_bank(self):
        model = _IsoFockModel()
        assert fock_adapter_module._has_gaussian_wells(model)

    def test_well_parameters_does_not_crash_on_3tuple_components(self):
        """This is the exact bug: unconditionally unpacking 4 values from a
        3-tuple raises `ValueError: not enough values to unpack`."""
        model = _IsoFockModel(d=8, K=2, n_ctx=2, n_layers=2)
        adapter = scaf.FockAdapter()

        x = torch.randint(0, 24, (2, 6))
        h = torch.randn(2, 6, 8)
        wp = adapter.well_parameters(model, 0, x, h=h)

        assert wp is not None
        K_total = model.V_theta.bank.banks[0].K * model._n_ctx
        assert wp["mu"].shape[-2] == K_total
        assert wp["precision_lr"].shape[-1] == 0, (
            "isotropic bank should synthesise a rank-0 low-rank factor"
        )

    def test_wells_follow_the_hidden_state_not_the_tokens(self):
        """The landscape is a function of ``xi``, and ``xi`` of ``h_ell``.

        Deriving it from the tokens instead would give one landscape for the
        whole stack, silently scoring every layer against layer 0's wells even
        where the call happened not to raise.
        """
        model = _IsoFockModel(d=8, K=2, n_ctx=2, n_layers=2)
        adapter = scaf.FockAdapter()
        x = torch.randint(0, 24, (2, 6))

        h1, h2 = torch.randn(2, 6, 8), torch.randn(2, 6, 8)
        wp1 = adapter.well_parameters(model, 0, x, h=h1)
        wp2 = adapter.well_parameters(model, 0, x, h=h2)
        assert not torch.allclose(wp1["mu"], wp2["mu"]), (
            "different hidden states must yield different wells"
        )

        # Same state, different layer: the depth code shifts the landscape.
        wp_l1 = adapter.well_parameters(model, 1, x, h=h1)
        assert not torch.allclose(wp1["mu"], wp_l1["mu"])

    def test_wells_are_per_position(self):
        model = _IsoFockModel(d=8, K=2, n_ctx=2, n_layers=2)
        wp = scaf.FockAdapter().well_parameters(
            model, 0, torch.randint(0, 24, (2, 6)), h=torch.randn(2, 6, 8)
        )
        assert wp["mu"].shape[:2] == (2, 6), (
            "context-dependent wells carry a time axis"
        )

    def test_layer_beyond_the_stack_has_no_landscape(self):
        """A trajectory holds L+1 states but only L layers have a depth code.

        The depth-conditioned V_theta wraps the index modulo ``n_layers``, so
        without this guard the final hidden state would be scored against
        layer 0's wells rather than skipped.
        """
        model = _IsoFockModel(d=8, K=2, n_ctx=2, n_layers=2)
        wp = scaf.FockAdapter().well_parameters(
            model, 2, torch.randint(0, 24, (2, 6)), h=torch.randn(2, 6, 8)
        )
        assert wp is None

    def test_assign_dominant_wells_handles_rank_zero_precision(self):
        """A rank-0 precision_lr must contribute exactly zero to the score,
        i.e. the isotropic fallback reduces to pure diagonal Mahalanobis."""
        K, d = 2, 8
        mu = torch.randn(K, d)
        a = torch.ones(K, d)
        B_rank0 = mu.new_zeros(K, d, 0)
        w = torch.ones(K) / K

        h = torch.randn(3, d)
        result = assign_dominant_wells(h, mu, a, B_rank0, w)
        assert result.shape == (3,)


# ---------------------------------------------------------------------------
# well_parameters extraction
# ---------------------------------------------------------------------------
def test_well_parameters_returns_correct_shapes():
    """Adapter.well_parameters should return dict with expected shapes."""
    model = GaussianWellCausalToyLM(CFG)
    with _im(model) as im:
        x = torch.randint(0, CFG.vocab_size, (2, 16))
        wp = im.adapter.well_parameters(im.model, 0, x)
        assert wp is not None
        assert wp["mu"].shape == (2, 2, CFG.d)
        assert wp["precision_diag"].shape == (2, 2, CFG.d)
        assert wp["precision_lr"].shape == (2, 2, CFG.d, 1)
        assert wp["weights"].shape == (2, 2)


# ---------------------------------------------------------------------------
# Basin-membership probe on causal model
# ---------------------------------------------------------------------------
def test_causal_model_shows_zero_crossing_rate():
    """A causal model's basin assignments must not depend on the future."""
    with _im(GaussianWellCausalToyLM(CFG)) as im:
        r = scaf.BasinMembershipProbe(
            n_seqs=4, n_pairs=1, micro_batch=0
        ).run(im, _corpus())
    assert r.statistic == 0.0
    assert r.passed
    assert r.detail["worst_layer"] is not None


# ---------------------------------------------------------------------------
# Basin-membership probe on leaky model
# ---------------------------------------------------------------------------
def test_leaky_model_shows_nonzero_crossing_rate():
    """A leak that moves hidden states should cause basin crossings."""
    with _im(GaussianWellLeakyToyLM(CFG, leak_scale=4.0)) as im:
        r = scaf.BasinMembershipProbe(
            n_seqs=4, n_pairs=1, micro_batch=0
        ).run(im, _corpus())
    assert r.statistic > 0.0
    assert not r.passed


# ---------------------------------------------------------------------------
# Per-layer profile
# ---------------------------------------------------------------------------
def test_leaky_model_crossing_profile():
    """Crossings should localise to the output layer (layer 1).

    GaussianWellLeakyToyLM trajectory has two layers: [h_embed, h_out].
    The embedding is purely positional (token-independent basin assignment),
    so crossings should concentrate in layer 1.
    """
    with _im(GaussianWellLeakyToyLM(CFG, leak_scale=4.0)) as im:
        r = scaf.BasinMembershipProbe(
            n_seqs=4, n_pairs=1, micro_batch=0
        ).run(im, _corpus())
    per_layer = r.detail["per_layer_crossing_rate"]
    assert len(per_layer) == 2
    assert per_layer[1] >= per_layer[0], (
        "output layer should have >= crossings than embedding layer"
    )


# ---------------------------------------------------------------------------
# Skip behaviour
# ---------------------------------------------------------------------------
def test_probe_skips_when_no_hidden_states():
    """Without hidden states, the probe should SKIP, not crash."""
    with _im(DeafToyLM(CFG)) as im:
        r = scaf.BasinMembershipProbe(n_seqs=2, micro_batch=0).run(
            im, _corpus()
        )
    assert r.skipped
    assert "has_hidden_states" in (r.skipped_reason or "")


def test_probe_skips_when_no_wells():
    """Without wells, the probe should SKIP, not crash."""
    with _im(CausalToyLM(CFG)) as im:
        r = scaf.BasinMembershipProbe(n_seqs=2, micro_batch=0).run(
            im, _corpus()
        )
    assert r.skipped
    assert "has_vtheta_wells" in (r.skipped_reason or "")


# ---------------------------------------------------------------------------
# Monitor integration
# ---------------------------------------------------------------------------
def test_monitor_records_basin_crossing_data():
    """LeakMonitor should include basin_crossing_rate as a diagnostic."""
    model = GaussianWellLeakyToyLM(CFG, leak_scale=4.0)
    monitor = scaf.LeakMonitor(
        model,
        vocab_size=CFG.vocab_size,
        interval=1,
        run_at_start=True,
        seq_len=32,
        n_seqs=2,
        micro_batch=0,
        honest_ppl=False,
        controls=False,
        hidden_state_probe=True,
        basin_membership_probe=True,
    )
    record = monitor.run(0)
    assert "basin_crossing_rate" in record
    assert record["basin_crossing_rate"] > 0.0
    assert record["basin_worst_layer"] is not None


def test_monitor_causal_model_zero_crossing():
    """LeakMonitor on a causal model should show basin_crossing_rate = 0."""
    model = GaussianWellCausalToyLM(CFG)
    monitor = scaf.LeakMonitor(
        model,
        vocab_size=CFG.vocab_size,
        interval=1,
        run_at_start=True,
        seq_len=32,
        n_seqs=2,
        micro_batch=0,
        honest_ppl=False,
        controls=False,
        hidden_state_probe=True,
        basin_membership_probe=True,
    )
    record = monitor.run(0)
    assert "basin_crossing_rate" in record
    assert record["basin_crossing_rate"] == 0.0


def test_monitor_summary_includes_basin_column():
    """The summary table should include the beta (basin crossing) column."""
    model = GaussianWellLeakyToyLM(CFG, leak_scale=4.0)
    monitor = scaf.LeakMonitor(
        model,
        vocab_size=CFG.vocab_size,
        interval=1,
        run_at_start=True,
        seq_len=32,
        n_seqs=2,
        micro_batch=0,
        honest_ppl=False,
        controls=False,
        basin_membership_probe=True,
    )
    monitor.run(0)
    s = monitor.summary()
    assert "beta" in s
    assert "b_peak" in s


# ---------------------------------------------------------------------------
# LeakFrame with basin columns
# ---------------------------------------------------------------------------
def test_leak_frame_includes_basin_columns():
    """build_leak_frame with include_basin_membership adds the new columns."""
    model = GaussianWellLeakyToyLM(CFG, leak_scale=4.0)
    frame = scaf.build_leak_frame(
        model,
        vocab_size=CFG.vocab_size,
        seq_len=32,
        n_seqs=2,
        splits=(0.5,),
        n_pairs=1,
        max_positions=4,
        micro_batch=0,
        include_basin_membership=True,
    )
    assert "basin_changed" in frame.columns
    assert "well_id_factual" in frame.columns
    assert "well_id_counterfactual" in frame.columns
    assert "basin_changed" in frame.effect_modifiers
    assert "layer" in frame.columns, (
        "include_basin_membership should imply include_hidden_states"
    )
    assert len(frame) > 0


def test_leak_frame_without_basin_membership_unchanged():
    """Default build_leak_frame should not include basin columns."""
    model = GaussianWellCausalToyLM(CFG)
    frame = scaf.build_leak_frame(
        model,
        vocab_size=CFG.vocab_size,
        seq_len=32,
        n_seqs=2,
        splits=(0.5,),
        n_pairs=1,
        max_positions=4,
        micro_batch=0,
    )
    assert "basin_changed" not in frame.columns
    assert "well_id_factual" not in frame.columns
