"""``LeakMonitor`` — scheduling, verdicts, and the non-interference guarantees.

Most of this file is about what the monitor must *not* do. A leak monitor is
only worth switching on if enabling it cannot change the run it observes and
cannot end it, so the RNG, dtype, training-mode, and hook-cleanup tests below
are load-bearing rather than defensive padding.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch
from torch import nn

import scaf
from tests.toy_models import (
    CausalToyLM,
    DeafToyLM,
    LeakyToyLM,
    PeekingToyLM,
    ToyConfig,
)

CFG = ToyConfig(vocab_size=24, d=16, max_len=64)
KW = dict(
    vocab_size=24, seq_len=32, n_seqs=4, n_targets=6, micro_batch=0,
    interval=100,
)


def monitor(model, **over):
    return scaf.LeakMonitor(model, **{**KW, **over})


# ----------------------------------------------------------------------
# Verdicts
# ----------------------------------------------------------------------
def test_causal_model_is_clean():
    r = monitor(CausalToyLM(CFG)).run(0)
    assert r["verdict"] == "CLEAN"
    assert r["linf"] == 0.0
    assert r["aile"] == 0.0


def test_leaky_model_is_flagged():
    r = monitor(LeakyToyLM(CFG, leak_scale=1.0)).run(0)
    assert r["verdict"] == "LEAK"
    assert r["aile"] > 0.0


def test_deaf_model_is_invalid_not_clean():
    """The blind-probe guard has to survive into the monitor.

    A model the probes cannot reach produces zero response everywhere, which
    looks exactly like a clean model. Certifying it would reproduce the
    original failure at every logged step instead of just once.
    """
    r = monitor(DeafToyLM(CFG)).run(0)
    assert r["verdict"] == "INVALID"
    assert r["controls_ok"] is False


def test_honest_ppl_stage_reports_the_nats_gap():
    r = monitor(PeekingToyLM(CFG)).run(0)
    assert r["tau_leak"] > 1.0
    assert r["ppl_honest"] > r["ppl_standard"]


def test_disabling_controls_yields_invalid_and_says_so():
    r = monitor(CausalToyLM(CFG), controls=False).run(0)
    assert r["verdict"] == "INVALID"
    assert "controls_ok" not in r


def test_honest_ppl_can_be_switched_off():
    r = monitor(CausalToyLM(CFG), honest_ppl=False).run(0)
    assert "tau_leak" not in r
    assert r["verdict"] == "CLEAN"


# ----------------------------------------------------------------------
# Scheduling
# ----------------------------------------------------------------------
def test_runs_at_start_then_on_interval():
    m = monitor(CausalToyLM(CFG), interval=100)
    fired = [s for s in (0, 50, 99, 100, 150, 201) if m.maybe_run(s)]
    assert fired == [0, 100, 201]


def test_run_at_start_false_anchors_the_schedule():
    m = monitor(CausalToyLM(CFG), interval=100, run_at_start=False)
    assert m.maybe_run(0) is None
    assert m.maybe_run(50) is None
    assert m.maybe_run(100) is not None


def test_schedule_survives_a_resume():
    """Resuming at a step that is not a multiple of the interval must still fire.

    Divisibility-based scheduling silently stops probing after a resume lands
    off the grid, which is the worst time to stop: a resumed run is exactly
    when you most want to know the leak state.
    """
    m = monitor(CausalToyLM(CFG), interval=1000, run_at_start=False)
    m.maybe_run(7000)
    assert m.maybe_run(7999) is None
    assert m.maybe_run(8000) is not None


def test_run_ignores_the_schedule():
    m = monitor(CausalToyLM(CFG), interval=10_000, run_at_start=False)
    assert m.run(5) is not None
    assert len(m.history) == 1


def test_rejects_a_non_positive_interval():
    with pytest.raises(ValueError, match="interval"):
        monitor(CausalToyLM(CFG), interval=0)


# ----------------------------------------------------------------------
# Non-interference
# ----------------------------------------------------------------------
def test_global_rng_is_restored():
    """The central guarantee: a monitored run must follow the same trajectory.

    If the monitor consumed RNG draws, every subsequent batch, dropout mask,
    and routing sample would shift, so turning monitoring on would change the
    result it is meant to observe.
    """
    m = monitor(CausalToyLM(CFG))
    torch.manual_seed(1234)
    expected = torch.randn(8)

    torch.manual_seed(1234)
    m.run(0)
    assert torch.equal(torch.randn(8), expected)


def test_training_mode_is_restored():
    model = CausalToyLM(CFG).train()
    monitor(model).run(0)
    assert model.training is True


def test_dtype_is_restored():
    model = CausalToyLM(CFG)
    before = next(model.parameters()).dtype
    monitor(model, dtype="float64").run(0)
    assert next(model.parameters()).dtype == before


def test_no_hooks_are_left_registered():
    """Hooks must not survive a measurement.

    A forward hook left installed runs on every training step afterwards —
    a cost the user did not ask for, and a path by which the auditor could
    perturb what it audits.
    """
    model = CausalToyLM(CFG)
    m = monitor(model)
    m.run(0)
    m.run(100)
    assert all(
        not mod._forward_hooks for mod in model.modules()
    )


def test_no_gradients_are_accumulated():
    """Probing must not pollute gradients a training loop is accumulating."""
    model = CausalToyLM(CFG)
    model(torch.zeros(1, 8, dtype=torch.long))[0].sum().backward()
    before = {n: p.grad.clone() for n, p in model.named_parameters()}
    monitor(model).run(0)
    for n, p in model.named_parameters():
        assert torch.equal(p.grad, before[n]), n


def test_repeated_runs_probe_identical_sequences():
    """A change in AILE must mean a change in the model, not in the draw."""
    m = monitor(LeakyToyLM(CFG, leak_scale=1.0))
    a, b = m.run(0), m.run(100)
    assert a["aile"] == b["aile"]
    assert a["linf"] == b["linf"]


# ----------------------------------------------------------------------
# Resilience
# ----------------------------------------------------------------------
class ExplodingLM(CausalToyLM):
    """Fails on any forward — stands in for an out-of-memory during a probe."""

    def forward(self, x):
        raise RuntimeError("CUDA out of memory (simulated)")


def test_a_failing_probe_does_not_kill_the_run():
    """A diagnostic must never be able to end a thousand-minute job."""
    m = monitor(ExplodingLM(CFG))
    r = m.run(0)
    assert r["verdict"] == "ERROR"
    assert "out of memory" in r["error"]
    assert len(m.history) == 1


def test_raise_on_leak_is_opt_in():
    assert monitor(PeekingToyLM(CFG)).run(0)["verdict"] == "LEAK"
    with pytest.raises(scaf.CausalLeakError, match="step 0"):
        monitor(PeekingToyLM(CFG), raise_on_leak=True).run(0)


def test_clean_model_never_raises_even_when_armed():
    assert monitor(CausalToyLM(CFG), raise_on_leak=True).run(0)["verdict"] == (
        "CLEAN"
    )


# ----------------------------------------------------------------------
# History and logging
# ----------------------------------------------------------------------
def test_first_leak_step_marks_where_the_log_stops_being_trustworthy():
    """The valve-opening story, simulated by opening the gate mid-run."""
    model = LeakyToyLM(CFG, leak_scale=0.0)
    m = monitor(model, honest_ppl=False)
    m.run(0)
    m.run(100)
    model.leak_scale = 1.0  # the optimiser discovers the channel
    m.run(200)
    assert m.first_leak_step == 200
    assert [r["verdict"] for r in m.history] == ["CLEAN", "CLEAN", "LEAK"]


def test_aile_trend_and_delta_track_the_valve_opening():
    model = LeakyToyLM(CFG, leak_scale=0.0)
    m = monitor(model, honest_ppl=False)
    m.run(0)
    model.leak_scale = 1.0
    r = m.run(100)
    assert r["aile_delta"] > 0
    assert [s for s, _ in m.aile_trend] == [0, 100]


def test_records_append_to_jsonl(tmp_path):
    path = tmp_path / "report.jsonl"
    m = monitor(CausalToyLM(CFG), jsonl_path=path)
    m.run(0)
    m.run(100)
    lines = path.read_text().strip().split("\n")
    assert len(lines) == 2
    assert json.loads(lines[0])["event"] == "scaf_leak_monitor"
    assert json.loads(lines[1])["step"] == 100


def test_summary_lists_every_measurement():
    m = monitor(PeekingToyLM(CFG))
    m.run(0)
    m.run(100)
    text = m.summary()
    assert "2 measurements" in text
    assert "first leak detected at step 0" in text


def test_summary_before_any_run_is_not_an_error():
    assert "no measurements" in monitor(CausalToyLM(CFG)).summary()


def test_scorecard_is_available_for_a_full_explanation():
    m = monitor(PeekingToyLM(CFG))
    m.run(0)
    assert m.last_scorecard.verdict == "LEAK"
    assert "VERDICT" in m.last_scorecard.summary()


# ----------------------------------------------------------------------
# Construction
# ----------------------------------------------------------------------
def test_accepts_real_tokens():
    tokens = np.random.default_rng(0).integers(0, CFG.vocab_size, size=4096)
    m = scaf.LeakMonitor(
        CausalToyLM(CFG), tokens=tokens,
        **{k: v for k, v in KW.items() if k != "vocab_size"},
    )
    assert m.corpus.__class__.__name__ == "TokenCorpus"
    assert m.run(0)["verdict"] == "CLEAN"


def test_needs_a_vocabulary_when_given_no_tokens():
    model = CausalToyLM(CFG)
    model.cfg = None
    with pytest.raises(ValueError, match="vocab_size"):
        scaf.LeakMonitor(model, seq_len=32)


def test_infers_the_models_device():
    model = CausalToyLM(CFG)
    assert monitor(model).device == next(model.parameters()).device


def test_model_without_parameters_defaults_to_cpu():
    m = scaf.LeakMonitor(nn.Module(), vocab_size=24, seq_len=32)
    assert m.device == torch.device("cpu")
