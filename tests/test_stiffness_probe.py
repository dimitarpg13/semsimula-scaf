"""Stiffness (Tier C) probe against a toy model with known curvature."""

from __future__ import annotations

import pytest
import torch

import scaf
from scaf.core.corpus import SyntheticCorpus
from scaf.probes.stiffness import (
    StiffnessProbe,
    _bootstrap_frac_ci,
    _frac,
    _quantiles,
    weyl_upper_bound,
)
from tests.toy_models import CausalToyLM, HarmonicToyLM, ToyConfig

CFG = ToyConfig(vocab_size=24, d=16, max_len=64, dt=1.0)


def _im(model, dtype=torch.float64):
    return scaf.InterventableModel(model, dtype=dtype)


def _corpus(seq_len=32, seed=0):
    return SyntheticCorpus(CFG.vocab_size, seq_len=seq_len, seed=seed)


# ---------------------------------------------------------------------------
# StiffnessProbe — skip-loudly behaviour
# ---------------------------------------------------------------------------
class TestSkipsLoudly:
    def test_skips_when_adapter_has_no_harmonic_terms(self):
        """CausalToyLM has no harmonic_terms; the probe must not fabricate a result."""
        with _im(CausalToyLM(CFG)) as im:
            assert not im.caps.has_harmonic_terms
            result = StiffnessProbe().run(im, _corpus())

        assert result.skipped
        assert result.passed is None
        assert "has_harmonic_terms" in result.skipped_reason


# ---------------------------------------------------------------------------
# StiffnessProbe — exact omega*dt values against a hand-known curvature
# ---------------------------------------------------------------------------
class TestExactStatistic:
    def test_stable_curvature_passes(self):
        """k_diag=0.25, dt=1.0 -> omega*dt = 0.5 everywhere: well below both bounds."""
        cfg = ToyConfig(vocab_size=24, d=8, max_len=64, dt=1.0)
        model = HarmonicToyLM(cfg, k_diag_value=0.25)

        with _im(model) as im:
            assert im.caps.has_harmonic_terms
            result = StiffnessProbe(n_batches=2, seqs_per_batch=2).run(
                im, SyntheticCorpus(cfg.vocab_size, seq_len=16, seed=0)
            )

        assert not result.skipped
        assert result.passed is True
        assert result.statistic == pytest.approx(0.0)
        assert result.detail["median"] == pytest.approx(0.5, abs=1e-6)
        assert result.detail["max"] == pytest.approx(0.5, abs=1e-6)
        assert result.detail["frac_marginal"] == pytest.approx(0.0)
        assert result.detail["dt"] == pytest.approx(1.0)

    def test_unstable_curvature_fails(self):
        """k_diag=9.0, dt=1.0 -> omega*dt = 3.0 everywhere: past the stability bound."""
        cfg = ToyConfig(vocab_size=24, d=8, max_len=64, dt=1.0)
        model = HarmonicToyLM(cfg, k_diag_value=9.0)

        with _im(model) as im:
            result = StiffnessProbe(n_batches=2, seqs_per_batch=2).run(
                im, SyntheticCorpus(cfg.vocab_size, seq_len=16, seed=0)
            )

        assert not result.skipped
        assert result.passed is False
        assert result.statistic == pytest.approx(1.0)
        assert result.detail["median"] == pytest.approx(3.0, abs=1e-6)
        assert result.detail["max"] == pytest.approx(3.0, abs=1e-6)
        assert result.detail["frac_marginal"] == pytest.approx(1.0)

    def test_marginal_band_and_dt_from_config(self):
        """k_diag=1.0, dt=1.5 -> omega*dt = 1.5: marginal, not unstable.

        Also exercises that dt is read from ModelAdapter.config()['dt'],
        not hardcoded.
        """
        cfg = ToyConfig(vocab_size=24, d=8, max_len=64, dt=1.5)
        model = HarmonicToyLM(cfg, k_diag_value=1.0)

        with _im(model) as im:
            assert im.config()["dt"] == pytest.approx(1.5)
            result = StiffnessProbe(n_batches=2, seqs_per_batch=2).run(
                im, SyntheticCorpus(cfg.vocab_size, seq_len=16, seed=0)
            )

        assert result.detail["dt"] == pytest.approx(1.5)
        assert result.detail["median"] == pytest.approx(1.5, abs=1e-6)
        assert result.detail["frac_marginal"] == pytest.approx(1.0)
        assert result.statistic == pytest.approx(0.0)  # frac_unstable
        assert result.passed is True

    def test_no_weyl_bound_without_vtheta_wells(self):
        """HarmonicToyLM has no well_parameters, so no eig_* keys are added."""
        cfg = ToyConfig(vocab_size=24, d=8, max_len=64, dt=1.0)
        model = HarmonicToyLM(cfg, k_diag_value=1.0)

        with _im(model) as im:
            assert not im.caps.has_vtheta_wells
            result = StiffnessProbe(n_batches=2, seqs_per_batch=2).run(
                im, SyntheticCorpus(cfg.vocab_size, seq_len=16, seed=0)
            )

        assert "eig_max" not in result.detail
        assert "eig_frac_unstable" not in result.detail


# ---------------------------------------------------------------------------
# StiffnessProbe — block-bootstrap CI
# ---------------------------------------------------------------------------
class TestBootstrapCi:
    def test_ci_present_with_multiple_batches(self):
        cfg = ToyConfig(vocab_size=24, d=8, max_len=64, dt=1.0)
        model = HarmonicToyLM(cfg, k_diag_value=1.0)

        with _im(model) as im:
            result = StiffnessProbe(n_batches=4, seqs_per_batch=2).run(
                im, SyntheticCorpus(cfg.vocab_size, seq_len=16, seed=0)
            )

        ci = result.detail["frac_unstable_ci95"]
        assert ci is not None
        lo, hi = ci
        assert 0.0 <= lo <= hi <= 1.0

    def test_ci_is_none_with_a_single_batch(self):
        cfg = ToyConfig(vocab_size=24, d=8, max_len=64, dt=1.0)
        model = HarmonicToyLM(cfg, k_diag_value=1.0)

        with _im(model) as im:
            result = StiffnessProbe(n_batches=1, seqs_per_batch=2).run(
                im, SyntheticCorpus(cfg.vocab_size, seq_len=16, seed=0)
            )

        assert result.detail["frac_unstable_ci95"] is None


# ---------------------------------------------------------------------------
# _bootstrap_frac_ci — direct unit tests
# ---------------------------------------------------------------------------
class TestBootstrapFracCiFunction:
    def test_none_with_fewer_than_two_blocks(self):
        assert _bootstrap_frac_ci([torch.rand(10)], threshold=2.0) is None
        assert _bootstrap_frac_ci([], threshold=2.0) is None

    def test_ci_brackets_the_true_rate(self):
        # Every block has exactly 50% of samples above the threshold, so the
        # CI should tightly bracket 0.5 regardless of resampling noise.
        blocks = [
            torch.tensor([0.0, 0.0, 3.0, 3.0]) for _ in range(20)
        ]
        ci = _bootstrap_frac_ci(blocks, threshold=2.0, n_boot=1000, seed=0)
        assert ci is not None
        lo, hi = ci
        assert lo <= 0.5 <= hi
        assert hi - lo < 0.2


# ---------------------------------------------------------------------------
# _quantiles / _frac — direct unit tests
# ---------------------------------------------------------------------------
class TestQuantilesAndFrac:
    def test_quantiles_of_constant_tensor(self):
        x = torch.full((100,), 0.75)
        q = _quantiles(x)
        for key in ("median", "p90", "p99", "p999", "max"):
            assert q[key] == pytest.approx(0.75)

    def test_empty_tensor_is_all_zero(self):
        q = _quantiles(torch.zeros(0))
        assert all(v == 0.0 for v in q.values())

    def test_frac_strict_greater_than(self):
        x = torch.tensor([1.0, 2.0, 3.0])
        assert _frac(x, 2.0) == pytest.approx(1.0 / 3.0)
        assert _frac(x, 0.5) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# weyl_upper_bound — direct unit tests against a hand-computed example
# ---------------------------------------------------------------------------
class TestWeylUpperBound:
    def test_matches_hand_computation_with_rank_one_correction(self):
        # One well (K=1), d=2, rank=1. a = [2, 3], B = [[1], [0]] so the
        # low-rank Gram matrix B^T B = [[1]] -> sigma_max^2 = 1. Evaluated
        # at the well centre, the Gaussian bump g = w * exp(0) = w = 1.
        mu = torch.zeros(1, 2)
        a = torch.tensor([[2.0, 3.0]])
        B = torch.tensor([[[1.0], [0.0]]])
        w = torch.tensor([1.0])
        h = torch.zeros(1, 2)

        k_weyl = weyl_upper_bound(
            h, {"mu": mu, "precision_diag": a, "precision_lr": B, "weights": w}
        )
        # K_Weyl = g * (max_i a_i + sigma_max(B)^2) = 1.0 * (3.0 + 1.0) = 4.0
        assert k_weyl.shape == (1,)
        assert k_weyl.item() == pytest.approx(4.0)

    def test_rank_zero_reduces_to_pure_diagonal(self):
        mu = torch.zeros(1, 2)
        a = torch.tensor([[2.0, 3.0]])
        B = torch.zeros(1, 2, 0)
        w = torch.tensor([1.0])
        h = torch.zeros(1, 2)

        k_weyl = weyl_upper_bound(
            h, {"mu": mu, "precision_diag": a, "precision_lr": B, "weights": w}
        )
        # No low-rank boost: K_Weyl = g * max_i a_i = 1.0 * 3.0 = 3.0
        assert k_weyl.item() == pytest.approx(3.0)

    def test_off_centre_bump_attenuates_the_bound(self):
        # Far from the well centre, the Gaussian bump g -> 0, so K_Weyl -> 0
        # regardless of how stiff the well itself is.
        mu = torch.zeros(1, 2)
        a = torch.tensor([[100.0, 100.0]])
        B = torch.zeros(1, 2, 0)
        w = torch.tensor([1.0])
        h = torch.full((1, 2), 10.0)

        k_weyl = weyl_upper_bound(
            h, {"mu": mu, "precision_diag": a, "precision_lr": B, "weights": w}
        )
        assert k_weyl.item() == pytest.approx(0.0, abs=1e-6)
