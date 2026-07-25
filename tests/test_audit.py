"""End-to-end ``scaf.audit`` behaviour and the scorecard's refusal semantics."""

from __future__ import annotations

import json

import numpy as np
import pytest

import scaf
from tests.toy_models import (
    CausalToyLM,
    DeafToyLM,
    LeakyToyLM,
    PeekingToyLM,
    PureLookaheadToyLM,
    ToyConfig,
)

CFG = ToyConfig(vocab_size=24, d=16, max_len=64)
KW = dict(dtype="float64", seq_len=32, n_seqs=4, n_targets=8, micro_batch=0)


def test_causal_model_is_certified_clean():
    report = scaf.audit(CausalToyLM(CFG), **KW)
    assert report.verdict == "CLEAN"
    assert report.passed
    report.assert_causal()


def test_leaky_model_is_flagged():
    report = scaf.audit(LeakyToyLM(CFG, leak_scale=1.0), **KW)
    assert report.verdict == "LEAK"
    with pytest.raises(scaf.CausalLeakError):
        report.assert_causal()


def test_deaf_model_is_invalid_not_clean():
    """The single most important assertion in the suite.

    A model the probes cannot reach must never be certified. If this test ever
    flips to CLEAN, SCAF has reproduced the exact failure it was built to
    prevent.
    """
    report = scaf.audit(DeafToyLM(CFG), **KW)
    assert report.verdict == "INVALID"
    assert not report.passed
    with pytest.raises(scaf.CausalLeakError):
        report.assert_causal()


def test_peeking_model_reports_the_honest_ppl_split():
    report = scaf.audit(PeekingToyLM(CFG), **KW)
    tr = report.get("target_relocation")
    assert tr.statistic > 1.0
    assert "honest PPL" in report.summary()
    assert report.verdict == "LEAK"


def test_purely_anticausal_model_is_invalid_not_leak():
    """Verdicts must not outrun the evidence the controls established.

    This model reads only the *next* token, so no intervention in the causal
    direction produces a response and the positive control cannot certify that
    the probes are live. ``INVALID`` is the honest answer even though the model
    is obviously broken.
    """
    report = scaf.audit(PureLookaheadToyLM(CFG), **KW)
    assert report.verdict == "INVALID"
    assert report.get("control_positive").passed is False


def test_audit_accepts_real_tokens():
    rng = np.random.default_rng(0)
    tokens = rng.integers(0, CFG.vocab_size, size=4096)
    report = scaf.audit(CausalToyLM(CFG), tokens=tokens, **KW)
    assert report.verdict == "CLEAN"
    # A real corpus removes the synthetic-corpus caveat.
    assert not any("synthetic corpus" in n for n in report.notes)


def test_synthetic_corpus_is_flagged_as_size_unreliable():
    report = scaf.audit(CausalToyLM(CFG), **KW)
    assert any("synthetic corpus" in n for n in report.notes)


def test_scorecard_serialises_for_jsonl_logging():
    report = scaf.audit(LeakyToyLM(CFG, leak_scale=1.0), **KW)
    payload = json.loads(report.to_json())
    assert payload["verdict"] == "LEAK"
    assert {c["name"] for c in payload["controls"]} == {
        "control_determinism", "control_placebo", "control_positive"
    }
    assert len(payload["probes"]) == 2
    assert "\n" not in report.to_json()


def test_audit_rejects_seq_len_over_max_len():
    with pytest.raises(ValueError, match="max_len"):
        scaf.audit(CausalToyLM(CFG), dtype="float64", seq_len=999, n_seqs=2)


def test_audit_needs_a_vocabulary_when_given_no_tokens():
    model = CausalToyLM(CFG)
    model.cfg = None  # a model that exposes no config at all
    with pytest.raises(ValueError, match="vocab_size"):
        scaf.audit(model, **KW)


def test_audit_restores_model_dtype():
    """An audit inside a training loop must not corrupt the model."""
    model = CausalToyLM(CFG)
    before = next(model.parameters()).dtype
    scaf.audit(model, **KW)
    assert next(model.parameters()).dtype == before
