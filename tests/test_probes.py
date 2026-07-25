"""Probe behaviour against models whose causal structure is known."""

from __future__ import annotations

import pytest
import torch

import scaf
from scaf.core.corpus import SyntheticCorpus
from tests.toy_models import (
    CausalToyLM,
    DeafToyLM,
    LeakyToyLM,
    PeekingToyLM,
    ToyConfig,
)

CFG = ToyConfig(vocab_size=24, d=16, max_len=64)


def _im(model):
    return scaf.InterventableModel(model, dtype=torch.float64)


def _corpus(seq_len=32, seed=0):
    return SyntheticCorpus(CFG.vocab_size, seq_len=seq_len, seed=seed)


# ---------------------------------------------------------------------------
# Future perturbation
# ---------------------------------------------------------------------------
def test_causal_model_shows_bit_exact_zero_leak():
    """The headline guarantee: causal means *exactly* zero, not 'small'."""
    with _im(CausalToyLM(CFG)) as im:
        r = scaf.FuturePerturbationProbe(n_seqs=4, micro_batch=0).run(
            im, _corpus()
        )
    assert r.statistic == 0.0
    assert r.detail["aile"] == 0.0
    assert r.passed


def test_leaky_model_is_detected():
    with _im(LeakyToyLM(CFG, leak_scale=1.0)) as im:
        r = scaf.FuturePerturbationProbe(n_seqs=4, micro_batch=0).run(
            im, _corpus()
        )
    assert r.statistic > 1e-6
    assert not r.passed


def test_leak_scale_zero_restores_causality():
    """The probe tracks the leak's magnitude, not merely the architecture."""
    with _im(LeakyToyLM(CFG, leak_scale=0.0)) as im:
        r = scaf.FuturePerturbationProbe(n_seqs=4, micro_batch=0).run(
            im, _corpus()
        )
    assert r.statistic == 0.0
    assert r.passed


def test_micro_batching_does_not_change_the_verdict():
    """Chunking is a memory optimisation and must be numerically inert."""
    model = LeakyToyLM(CFG, leak_scale=1.0)
    with _im(model) as im:
        whole = scaf.FuturePerturbationProbe(
            n_seqs=4, micro_batch=0
        ).run(im, _corpus(seed=7))
        chunked = scaf.FuturePerturbationProbe(
            n_seqs=4, micro_batch=2
        ).run(im, _corpus(seed=7))
    assert whole.statistic == pytest.approx(chunked.statistic, rel=1e-9)


# ---------------------------------------------------------------------------
# Target relocation / honest PPL
# ---------------------------------------------------------------------------
def test_causal_model_has_no_honest_ppl_gap():
    with _im(CausalToyLM(CFG)) as im:
        r = scaf.TargetRelocationProbe(
            n_seqs=4, n_targets=8, micro_batch=0
        ).run(im, _corpus())
    assert abs(r.statistic) < 1e-9
    assert r.passed
    assert r.detail["ppl_standard"] == pytest.approx(
        r.detail["ppl_honest"], rel=1e-6
    )


def test_peeking_model_shows_large_honest_ppl_gap():
    """Miniature of the d=384 finding: excellent PPL, terrible honest PPL."""
    with _im(PeekingToyLM(CFG)) as im:
        r = scaf.TargetRelocationProbe(
            n_seqs=4, n_targets=8, micro_batch=0
        ).run(im, _corpus())
    assert r.statistic > 1.0, "expected a multi-nat leak tax"
    assert not r.passed
    d = r.detail
    assert d["ppl_standard"] < 2.0
    assert d["ppl_honest"] > 5.0 * d["ppl_standard"]


def test_target_relocation_skips_loudly_on_short_sequences():
    """A probe that cannot run must say so, never silently pass."""
    tiny = SyntheticCorpus(CFG.vocab_size, seq_len=4)
    with _im(CausalToyLM(CFG)) as im:
        r = scaf.TargetRelocationProbe(n_seqs=2, micro_batch=0).run(im, tiny)
    assert r.skipped
    assert r.passed is None
    assert r.skipped_reason


# ---------------------------------------------------------------------------
# Controls
# ---------------------------------------------------------------------------
def test_controls_pass_on_a_well_behaved_model():
    with _im(CausalToyLM(CFG)) as im:
        c = _corpus()
        assert scaf.DeterminismControl(n_seqs=2).run(im, c).passed
        assert scaf.PlaceboControl(n_seqs=2).run(im, c).passed
        assert scaf.PositiveControl(n_seqs=2).run(im, c).passed


def test_positive_control_fails_on_a_deaf_model():
    """The anti-blind-probe guarantee.

    ``DeafToyLM`` ignores its input, so the leak probes see nothing and would
    happily report 'clean'. The positive control is the only thing standing
    between that and a false certification.
    """
    with _im(DeafToyLM(CFG)) as im:
        c = _corpus()
        leak = scaf.FuturePerturbationProbe(n_seqs=2, micro_batch=0).run(im, c)
        pos = scaf.PositiveControl(n_seqs=2).run(im, c)

    assert leak.statistic == 0.0 and leak.passed, "leak probe is fooled"
    assert not pos.passed, "positive control must catch the blind probe"
