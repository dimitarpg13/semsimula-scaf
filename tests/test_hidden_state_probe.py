"""Hidden-state cosine-deviation probe against models with known structure."""

from __future__ import annotations

import pytest
import torch

import scaf
from scaf.core.corpus import SyntheticCorpus
from tests.toy_models import (
    CausalToyLM,
    FockLikeToyLM,
    LeakyToyLM,
    ToyConfig,
)

CFG = ToyConfig(vocab_size=24, d=16, max_len=64)


def _im(model):
    return scaf.InterventableModel(model, dtype=torch.float64)


def _corpus(seq_len=32, seed=0):
    return SyntheticCorpus(CFG.vocab_size, seq_len=seq_len, seed=seed)


# ---------------------------------------------------------------------------
# Trajectory access
# ---------------------------------------------------------------------------
def test_causal_model_exposes_hidden_states():
    """Adapter reports has_hidden_states when the model supports trajectories."""
    with _im(CausalToyLM(CFG)) as im:
        assert im.caps.has_hidden_states
        logits, traj = im.batch_logits_with_trajectory(
            torch.randint(0, CFG.vocab_size, (2, 16))
        )
        assert logits.shape[0] == 2
        assert len(traj) >= 2
        for h in traj:
            assert h.shape[:2] == (2, 16)


# ---------------------------------------------------------------------------
# Hidden-state probe on causal model
# ---------------------------------------------------------------------------
def test_causal_model_shows_zero_cosine_deviation():
    """The hidden-state analogue of the linf=0 guarantee.

    Unlike logit L∞ (which is bit-exact zero for a causal model because the
    logit tensors are the same object), cosine similarity involves a division
    that introduces rounding at machine epsilon. The threshold must absorb
    this, which is why HiddenStateLeakProbe accepts a tolerance.
    """
    EPS = 1e-12
    with _im(CausalToyLM(CFG)) as im:
        r = scaf.HiddenStateLeakProbe(
            n_seqs=4, n_pairs=1, threshold=EPS, micro_batch=0
        ).run(im, _corpus())
    assert r.statistic < EPS, f"expected near-zero, got {r.statistic}"
    assert r.passed
    assert not r.detail["latent_leak"]
    for layer_dev in r.detail["per_layer_delta_cos"]:
        assert layer_dev < EPS


# ---------------------------------------------------------------------------
# Hidden-state probe on leaky model
# ---------------------------------------------------------------------------
def test_leaky_model_shows_nonzero_cosine_deviation():
    """A global-pool leak corrupts hidden states before it corrupts logits."""
    with _im(LeakyToyLM(CFG, leak_scale=1.0)) as im:
        r = scaf.HiddenStateLeakProbe(
            n_seqs=4, n_pairs=1, micro_batch=0
        ).run(im, _corpus())
    assert r.statistic > 0.0
    assert not r.passed


def test_leak_scale_zero_restores_hidden_state_causality():
    """Hidden-state probe tracks the leak magnitude, not the architecture."""
    EPS = 1e-12
    with _im(LeakyToyLM(CFG, leak_scale=0.0)) as im:
        r = scaf.HiddenStateLeakProbe(
            n_seqs=4, n_pairs=1, threshold=EPS, micro_batch=0
        ).run(im, _corpus())
    assert r.statistic < EPS
    assert r.passed


# ---------------------------------------------------------------------------
# Fock-like model: leak localises to the output layer
# ---------------------------------------------------------------------------
def test_fock_like_leak_shows_per_layer_profile():
    """The per-layer profile should show the leak in the output layer.

    FockLikeToyLM has two trajectory layers: [h_embed, h_out]. The embedding
    layer h_0 = emb(x) is purely causal (each position embeds independently).
    The output layer h_1 adds the global-pool leak. The probe should detect
    deviation in h_1 but not in h_0.
    """
    EPS = 1e-12
    with _im(FockLikeToyLM(CFG, leak_scale=1.0, gate_init=1.0)) as im:
        r = scaf.HiddenStateLeakProbe(
            n_seqs=4, n_pairs=1, micro_batch=0
        ).run(im, _corpus())
    per_layer = r.detail["per_layer_delta_cos"]
    assert len(per_layer) == 2, f"expected 2 layers, got {len(per_layer)}"
    assert per_layer[0] < EPS, "embedding layer must be causal"
    assert per_layer[1] > 1e-3, "output layer should carry a substantial leak"
    assert r.detail["peak_layer"] == 1


def test_fock_like_gate_zero_restores_hidden_state_causality():
    """Closing the reverse-channel gate must restore hidden-state causality."""
    EPS = 1e-12
    with _im(FockLikeToyLM(CFG, leak_scale=1.0, gate_init=0.0)) as im:
        r = scaf.HiddenStateLeakProbe(
            n_seqs=4, n_pairs=1, threshold=EPS, micro_batch=0
        ).run(im, _corpus())
    assert r.statistic < EPS
    assert r.passed


# ---------------------------------------------------------------------------
# Micro-batching invariance
# ---------------------------------------------------------------------------
def test_micro_batching_does_not_change_hidden_state_verdict():
    """Chunking must be numerically inert for the hidden-state probe too."""
    model = LeakyToyLM(CFG, leak_scale=1.0)
    with _im(model) as im:
        whole = scaf.HiddenStateLeakProbe(
            n_seqs=4, n_pairs=1, micro_batch=0
        ).run(im, _corpus(seed=7))
        chunked = scaf.HiddenStateLeakProbe(
            n_seqs=4, n_pairs=1, micro_batch=2
        ).run(im, _corpus(seed=7))
    assert whole.statistic == pytest.approx(chunked.statistic, rel=1e-9)


# ---------------------------------------------------------------------------
# Skip behaviour
# ---------------------------------------------------------------------------
def test_probe_skips_when_no_hidden_state_support():
    """An adapter without has_hidden_states should produce a SKIP, not crash."""
    from tests.toy_models import DeafToyLM

    with _im(DeafToyLM(CFG)) as im:
        r = scaf.HiddenStateLeakProbe(n_seqs=2, micro_batch=0).run(
            im, _corpus()
        )
    assert r.skipped
    assert r.passed is None
    assert "has_hidden_states" in (r.skipped_reason or "")


# ---------------------------------------------------------------------------
# Monitor integration
# ---------------------------------------------------------------------------
def test_monitor_records_hidden_state_data():
    """LeakMonitor should include max_cos_dev as a diagnostic.

    The hidden-state probe is a diagnostic instrument (not a verdict gate),
    so its result is reported alongside the verdict without changing it.
    The monitor runs at float32 by default, so the embedding layer may show
    noise up to ~1e-6 from cosine computation.
    """
    model = FockLikeToyLM(CFG, leak_scale=1.0, gate_init=1.0)
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
    )
    record = monitor.run(0)
    assert "max_cos_dev" in record
    assert record["max_cos_dev"] > 0.0
    assert record["peak_layer"] == 1
    assert record["per_layer_delta_cos"][0] < 1e-5, (
        "embedding layer should be causal (noise only at float32 precision)"
    )


def test_monitor_summary_includes_cosine_column():
    """The summary table should include the cos_dev column."""
    model = FockLikeToyLM(CFG, leak_scale=1.0, gate_init=1.0)
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
    )
    monitor.run(0)
    s = monitor.summary()
    assert "cos_dev" in s
    assert "peak_L" in s


# ---------------------------------------------------------------------------
# LeakFrame with hidden-state columns
# ---------------------------------------------------------------------------
def test_leak_frame_includes_hidden_state_columns():
    """build_leak_frame with include_hidden_states adds the new columns."""
    model = FockLikeToyLM(CFG, leak_scale=1.0, gate_init=1.0)
    frame = scaf.build_leak_frame(
        model,
        vocab_size=CFG.vocab_size,
        seq_len=32,
        n_seqs=2,
        splits=(0.5,),
        n_pairs=1,
        max_positions=4,
        micro_batch=0,
        include_hidden_states=True,
    )
    assert "hidden_cos_dev" in frame.columns
    assert "layer" in frame.columns
    assert "layer" in frame.effect_modifiers
    assert len(frame) > 0
    # Every row should have a layer value
    for layer_val in frame.column("layer"):
        assert isinstance(layer_val, int)
        assert layer_val >= 0


def test_leak_frame_without_hidden_states_unchanged():
    """Default build_leak_frame should not include hidden-state columns."""
    model = FockLikeToyLM(CFG, leak_scale=1.0, gate_init=1.0)
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
    assert "hidden_cos_dev" not in frame.columns
    assert "layer" not in frame.columns
