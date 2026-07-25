"""Mediation: attributing a leak to the component that carries it."""

from __future__ import annotations

import math

import pytest
import torch

import scaf
from scaf.core.corpus import SyntheticCorpus
from tests.toy_models import (
    CausalToyLM,
    FockLikeToyLM,
    LeakyToyLM,
    ToyConfig,
    TwoChannelLeakToyLM,
)

CFG = ToyConfig(vocab_size=24, d=16, max_len=64)


def _im(model):
    return scaf.InterventableModel(model, dtype=torch.float64)


def _corpus(seed=0):
    return SyntheticCorpus(CFG.vocab_size, seq_len=32, seed=seed)


def _probe(**kw):
    return scaf.MediationProbe(n_seqs=4, micro_batch=0, **kw)


# ---------------------------------------------------------------------------
# Attribution
# ---------------------------------------------------------------------------
def test_sole_carrier_is_fully_attributed():
    """The Fock case in miniature: one gate carries the entire leak."""
    with _im(FockLikeToyLM(CFG)) as im:
        r = _probe().run(im, _corpus())

    assert r.detail["top_mediator"] == "reverse_channel_scale"
    assert r.statistic == pytest.approx(1.0, abs=1e-9)
    assert r.detail["cde_reverse_channel_scale"] == 0.0
    assert r.passed


def test_two_carriers_each_get_partial_attribution():
    """Neither carrier alone explains a two-channel leak."""
    with _im(TwoChannelLeakToyLM(CFG, major=1.0, minor=0.25)) as im:
        r = _probe().run(im, _corpus())

    assert {"reverse_channel_scale", "exchange_scale"} <= set(
        r.detail["ranking"]
    )
    for name in ("reverse_channel_scale", "exchange_scale"):
        frac = r.detail[f"attributed_{name}"]
        assert 0.0 < frac < 1.0, f"{name} should be a partial carrier"
        # Leak survives either single knockout.
        assert r.detail[f"cde_{name}"] > 0.0
    assert not r.passed


@pytest.mark.parametrize("major, minor", [(1.0, 0.25), (2.0, 0.05)])
def test_attribution_is_calibrated_against_the_known_gate_share(major, minor):
    """Attribution should recover the share the gates actually set.

    Both channels read the same pool, so channel strength is ``tanh(gate)`` and
    the dominant channel's analytic share is
    ``tanh(major) / (tanh(major) + tanh(minor))``. Checking the measured value
    against that number tests calibration, not merely ordering — a probe could
    rank correctly while reporting meaningless magnitudes.

    The tolerance is loose because the two channels add as vectors, so the
    measured shares are sub-additive rather than summing to one.
    """
    analytic = math.tanh(major) / (math.tanh(major) + math.tanh(minor))
    with _im(TwoChannelLeakToyLM(CFG, major=major, minor=minor)) as im:
        r = _probe().run(im, _corpus())
    assert r.detail["attributed_reverse_channel_scale"] == pytest.approx(
        analytic, abs=0.05
    )


@pytest.mark.parametrize(
    "major, minor, expected",
    [
        (2.0, 0.05, "reverse_channel_scale"),
        (0.05, 2.0, "exchange_scale"),
    ],
)
def test_ranking_follows_measured_leak_strength(major, minor, expected):
    """The ranking must track the causal contribution, not the declaration order.

    Asserting a fixed winner would be circular — ``reverse_channel_scale`` is
    listed first by the adapter, so a probe that ignored the measurement
    entirely would still "pass". Flipping which gate is open and requiring the
    ranking to flip with it is what actually tests the attribution.
    """
    model = TwoChannelLeakToyLM(CFG, major=major, minor=minor)
    with _im(model) as im:
        r = _probe().run(im, _corpus())
    assert r.detail["top_mediator"] == expected


def test_attribution_uses_identical_counterfactuals_across_arms():
    """Total and CDE must be comparable, so the probe is fully reproducible."""
    with _im(TwoChannelLeakToyLM(CFG)) as im:
        a = _probe().run(im, _corpus(seed=5))
        b = _probe().run(im, _corpus(seed=5))
    assert a.statistic == pytest.approx(b.statistic, rel=1e-12)
    assert a.detail["total"] == pytest.approx(b.detail["total"], rel=1e-12)


def test_attribution_is_computed_on_the_mean_not_the_max():
    """Attribution must use the mean effect; the max is unstable for it.

    ``linf`` is a max over positions, so two partially-cancelling channels can
    make removing the weaker one *raise* it, producing a spurious negative
    share. The mean aggregates over all positions and is near-additive, which
    is also what the CDE's definition as an expectation calls for.
    """
    with _im(TwoChannelLeakToyLM(CFG, major=1.0, minor=0.25)) as im:
        r = _probe().run(im, _corpus())
    assert "total" in r.detail and "total_linf" in r.detail
    # Shares are taken against the mean effect, not the max.
    name = r.detail["top_mediator"]
    expected = 1.0 - r.detail[f"cde_{name}"] / r.detail["total"]
    assert r.statistic == pytest.approx(expected, rel=1e-12)


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------
def test_mediation_skips_when_there_is_no_leak():
    """Attributing a zero total would turn rounding noise into percentages."""
    with _im(FockLikeToyLM(CFG, gate_init=0.0)) as im:
        r = _probe().run(im, _corpus())
    assert r.skipped
    assert "no leak to attribute" in r.skipped_reason


def test_mediation_skips_when_no_mediators_are_declared():
    with _im(LeakyToyLM(CFG, leak_scale=1.0)) as im:
        r = _probe().run(im, _corpus())
    assert r.skipped
    assert "no mediators" in r.skipped_reason


def test_explicit_unknown_mediator_is_reported_not_intervenable():
    with _im(FockLikeToyLM(CFG)) as im:
        r = _probe(
            mediators=("reverse_channel_scale", "not_a_real_component")
        ).run(im, _corpus())
    assert r.detail["not_intervenable"] == ["not_a_real_component"]
    assert r.detail["top_mediator"] == "reverse_channel_scale"


def test_ineffective_knockout_is_flagged_rather_than_read_as_innocence():
    """A knockout that changes nothing means the clamp reference is wrong.

    ``creation_gate`` here is an ``nn.Identity`` sitting outside the leak path,
    so zeroing its output cannot reduce the leak. The probe must say the
    knockout was not verified rather than imply the component is exonerated.
    """
    with _im(FockLikeToyLM(CFG)) as im:
        r = _probe(mediators=("creation_gate",)).run(im, _corpus())
    assert r.detail["attributed_creation_gate"] == pytest.approx(0.0, abs=1e-12)
    assert r.detail["knockout_verified"] is False
    assert not r.passed


# ---------------------------------------------------------------------------
# Integration with audit()
# ---------------------------------------------------------------------------
def test_audit_attributes_a_leak_without_changing_the_verdict():
    report = scaf.audit(
        FockLikeToyLM(CFG), dtype="float64", seq_len=32,
        n_seqs=4, n_targets=8, micro_batch=0,
    )
    assert report.verdict == "LEAK"
    med = report.get("mediation")
    assert med is not None
    assert med.detail["top_mediator"] == "reverse_channel_scale"
    assert "leak attribution" in report.summary()


def test_diagnostics_never_affect_the_verdict():
    """A well-attributed leak is still a leak.

    Guards against attribution quality leaking into the causality verdict,
    which would let a thoroughly-explained leak read as healthier than an
    unexplained one.
    """
    report = scaf.audit(
        FockLikeToyLM(CFG), dtype="float64", seq_len=32,
        n_seqs=4, n_targets=8, micro_batch=0,
    )
    assert report.get("mediation").passed is True
    assert report.verdict == "LEAK"
    with pytest.raises(scaf.CausalLeakError):
        report.assert_causal()


def test_clean_model_skips_mediation_entirely():
    """Attribution must cost nothing on a model with no leak."""
    report = scaf.audit(
        CausalToyLM(CFG), dtype="float64", seq_len=32,
        n_seqs=4, n_targets=8, micro_batch=0,
    )
    assert report.verdict == "CLEAN"
    assert report.diagnostics == []


def test_mediation_can_be_disabled():
    report = scaf.audit(
        FockLikeToyLM(CFG), dtype="float64", seq_len=32,
        n_seqs=4, n_targets=8, micro_batch=0, mediation=False,
    )
    assert report.verdict == "LEAK"
    assert report.diagnostics == []


def test_diagnostics_serialise_into_the_jsonl_payload():
    import json

    report = scaf.audit(
        FockLikeToyLM(CFG), dtype="float64", seq_len=32,
        n_seqs=4, n_targets=8, micro_batch=0,
    )
    payload = json.loads(report.to_json())
    assert payload["diagnostics"][0]["name"] == "mediation"
    assert "\n" not in report.to_json()


def test_model_is_restored_after_every_knockout():
    """An audit inside a training loop must not leave a clamped gate behind."""
    model = FockLikeToyLM(CFG)
    before = model.reverse_channel_scale.detach().clone()
    scaf.audit(
        model, dtype="float64", seq_len=32,
        n_seqs=4, n_targets=8, micro_batch=0,
    )
    assert torch.allclose(model.reverse_channel_scale, before)
