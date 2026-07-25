"""Adapter resolution, the intervention surface, and corpus mechanics."""

from __future__ import annotations

import pytest
import torch

import scaf
from scaf.core.corpus import SyntheticCorpus, TokenCorpus
from tests.toy_models import CausalToyLM, FockLikeToyLM, ToyConfig

CFG = ToyConfig(vocab_size=24, d=16, max_len=64)


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------
def test_fock_adapter_wins_on_structural_markers():
    """Detection is duck-typed, so it works with no research repo installed."""
    assert scaf.resolve_adapter(FockLikeToyLM(CFG)).name == "fock"


def test_generic_adapter_is_the_fallback():
    assert scaf.resolve_adapter(CausalToyLM(CFG)).name == "generic"


def test_explicit_adapter_overrides_detection():
    a = scaf.resolve_adapter(FockLikeToyLM(CFG), adapter=scaf.GenericAdapter())
    assert a.name == "generic"


def test_model_supplied_hook_takes_precedence_over_the_registry():
    model = CausalToyLM(CFG)
    model.__scaf_adapter__ = lambda: scaf.FockAdapter()
    assert scaf.resolve_adapter(model).name == "fock"


def test_non_module_is_rejected():
    with pytest.raises(TypeError):
        scaf.resolve_adapter("not a model")


def test_registry_is_ordered_by_descending_priority():
    prios = [c.priority for c in scaf.registered_adapters()]
    assert prios == sorted(prios, reverse=True)


# ---------------------------------------------------------------------------
# Intervention surface
# ---------------------------------------------------------------------------
def test_fock_capabilities_expose_the_reverse_channel():
    with scaf.InterventableModel(FockLikeToyLM(CFG)) as im:
        assert im.caps.has_reverse_channel
        assert im.caps.has_registers
        assert "reverse_channel_scale" in im.clampable_names
        assert "reverse_ch" in im.intervention_names


def test_fock_adapter_declares_the_exchange_gate_too():
    """``FockAttentionPARFLM`` gates a direct exchange force with
    ``torch.tanh(exchange_scale)`` — structurally the same kind of
    non-conservative channel as the reverse channel, and just as able to carry
    a leak. If the adapter omits it, mediation reports a confident attribution
    having never tested one of the two candidate carriers.
    """
    from tests.toy_models import TwoChannelLeakToyLM

    with scaf.InterventableModel(TwoChannelLeakToyLM(CFG)) as im:
        assert im.adapter.name == "fock"
        assert "exchange_scale" in im.clampable_names
        assert "reverse_channel_scale" in im.clampable_names
        assert "exchange_scale" in im.mediators


def test_unknown_intervention_name_raises():
    """Silently ignoring a knockout would make a leaky model look clean."""
    with scaf.InterventableModel(FockLikeToyLM(CFG)) as im:
        with pytest.raises(KeyError, match="nonexistent"):
            with im.do(nonexistent=lambda t: t):
                pass
        with pytest.raises(KeyError, match="nonexistent"):
            with im.clamp(nonexistent=0.0):
                pass


def test_clamp_restores_the_original_value():
    model = FockLikeToyLM(CFG)
    with scaf.InterventableModel(model) as im:
        original = model.reverse_channel_scale.detach().clone()
        with im.clamp(reverse_channel_scale=5.0):
            assert torch.allclose(
                model.reverse_channel_scale,
                torch.full_like(model.reverse_channel_scale, 5.0),
            )
        assert torch.allclose(model.reverse_channel_scale, original)


def test_knocking_out_the_mediator_removes_the_leak():
    """Mediation in miniature: the gate is the leak's sole carrier.

    ``knockout`` clamps the gate parameter to zero, and ``tanh(0) = 0`` closes
    the channel — the same arithmetic as the real models. This is the toy
    analogue of showing that the reverse channel, not the registers
    themselves, is what carries the Fock leak.
    """
    corpus = SyntheticCorpus(CFG.vocab_size, seq_len=32)
    probe = scaf.FuturePerturbationProbe(n_seqs=4, micro_batch=0)

    with scaf.InterventableModel(FockLikeToyLM(CFG), dtype=torch.float64) as im:
        leaking = probe.run(im, corpus)
        with im.knockout("reverse_channel_scale"):
            knocked_out = probe.run(im, corpus)

    assert leaking.statistic > 1e-6
    assert knocked_out.statistic == 0.0


def test_a_nonzero_gate_value_does_not_count_as_a_knockout():
    """Guards the arithmetic that makes ``knockout`` meaningful.

    ``tanh`` is odd, so driving the gate far *negative* flips the leak's sign
    while leaving its magnitude at full strength. Only zero closes the channel.
    A knockout implemented as "push the parameter to an extreme" would look
    like it worked while the leak ran on untouched.
    """
    corpus = SyntheticCorpus(CFG.vocab_size, seq_len=32)
    probe = scaf.FuturePerturbationProbe(n_seqs=4, micro_batch=0)

    with scaf.InterventableModel(FockLikeToyLM(CFG), dtype=torch.float64) as im:
        with im.clamp(reverse_channel_scale=-60.0):
            still_leaking = probe.run(im, corpus)

    assert still_leaking.statistic > 1e-6


def test_hooks_are_removed_on_close():
    model = FockLikeToyLM(CFG)
    im = scaf.InterventableModel(model)
    im.close()
    assert not im._handles
    assert not model.reverse_ch._forward_hooks


# ---------------------------------------------------------------------------
# Corpus
# ---------------------------------------------------------------------------
def test_perturb_suffix_preserves_the_prefix_exactly():
    c = SyntheticCorpus(CFG.vocab_size, seq_len=32, seed=1)
    x = c.sample(4)
    y = c.perturb_suffix(x, t_p=15)
    assert torch.equal(x[:, :16], y[:, :16])
    assert not torch.equal(x[:, 16:], y[:, 16:])


def test_perturb_suffix_rejects_an_out_of_range_split():
    c = SyntheticCorpus(CFG.vocab_size, seq_len=32)
    with pytest.raises(ValueError):
        c.perturb_suffix(c.sample(2), t_p=31)


def test_perturb_position_changes_exactly_one_column():
    c = SyntheticCorpus(CFG.vocab_size, seq_len=32, seed=2)
    x = c.sample(4)
    y = c.perturb_position(x, s=10)
    differing = (x != y).any(dim=0)
    assert differing[10]
    assert differing.sum() == 1


def test_corpus_rng_is_private_and_reproducible():
    """An audit must not be perturbed by the caller's global seed."""
    a = SyntheticCorpus(CFG.vocab_size, seq_len=16, seed=3)
    torch.manual_seed(999)
    first = a.sample(2)
    a.reset()
    torch.manual_seed(1)
    assert torch.equal(first, a.sample(2))


def test_token_corpus_resamples_from_observed_tokens():
    """Perturbations must stay in-distribution to size a leak honestly."""
    tokens = [3, 4, 5] * 200
    c = TokenCorpus(tokens, seq_len=16, seed=0)
    y = c.perturb_suffix(c.sample(4), t_p=7)
    assert set(y.flatten().tolist()) <= {3, 4, 5}


def test_token_corpus_rejects_too_little_data():
    with pytest.raises(ValueError, match="at least"):
        TokenCorpus([1, 2, 3], seq_len=32)


@pytest.mark.parametrize("kind", ["list", "torch", "torch_2d", "numpy"])
def test_token_corpus_accepts_every_common_container(kind):
    """Guards the torch/NumPy bridge.

    Passing a torch tensor is the most natural notebook usage, and a NumPy
    array the second most. Neither may route through ``Tensor.numpy()`` or
    ``torch.from_numpy``, which fail when torch and NumPy major versions
    disagree.
    """
    raw = list(range(20)) * 10
    if kind == "list":
        tokens = raw
    elif kind == "torch":
        tokens = torch.tensor(raw)
    elif kind == "torch_2d":
        tokens = torch.tensor(raw).reshape(10, 20)
    else:
        np = pytest.importorskip("numpy")
        tokens = np.array(raw, dtype="int64")

    c = TokenCorpus(tokens, seq_len=16, seed=0)
    x = c.sample(3)
    assert x.shape == (3, 16)
    assert x.dtype == torch.long
    assert c.vocab_size == 20


def test_corpus_sampling_is_seed_reproducible():
    a = TokenCorpus(list(range(50)) * 10, seq_len=16, seed=11)
    b = TokenCorpus(list(range(50)) * 10, seq_len=16, seed=11)
    assert torch.equal(a.sample(4), b.sample(4))
