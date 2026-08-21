"""An intervention point must be *reached*, not merely *registered*.

A forward hook fires only when a module is invoked through ``__call__``. Three
constructs break that assumption, and all three occur in the production Fock
stack:

* a module held in an ``nn.ModuleList``, whose container ``forward`` is never
  called (``destruction_gates[l](r)``),
* a module invoked through a named method (``V_phi.forward_gathered``,
  ``creation_gate_qkv.forward_prefix``, ``V_theta.analytical_grad``), chosen by
  a config flag,
* a module that the current configuration simply never reaches.

In each case ``do()`` used to install a hook that no forward would ever
trigger, so the block measured the *unmodified* model while reporting an
intervention — a knockout scored at zero effect, indistinguishable from a
component that carries none of the leak. That is a false all-clear, which is
what axiom A7 exists to prevent, so these tests assert the failure is loud.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn
from toy_models import FockLikeToyLM, ToyConfig

import scaf
from scaf.core.intervenable import InertIntervention, zero_out

CFG = ToyConfig()


class _GatedFockToy(FockLikeToyLM):
    """Adds the three awkward invocation patterns to the Fock-like toy.

    ``destruction_gates`` is a ``ModuleList`` applied per layer, ``V_phi`` is
    reached only through ``forward_gathered``, and ``creation_gates`` is
    declared but never called.
    """

    def __init__(self, cfg: ToyConfig | None = None, n_layers: int = 3):
        super().__init__(cfg)
        d = self.cfg.d
        self.destruction_gates = nn.ModuleList(
            nn.Linear(d, d) for _ in range(n_layers)
        )
        self.creation_gates = nn.ModuleList(nn.Linear(d, d) for _ in range(2))
        self.V_phi = _GatheredOnly(d)

    def forward(self, x):
        h = self._mix(self.emb(x))
        for gate in self.destruction_gates:
            h = h + torch.tanh(gate(h))
        h = h + self.V_phi.forward_gathered(h)
        return self.out(h), None


class _GatheredOnly(nn.Module):
    """Reached only through ``forward_gathered``; ``forward`` raises if used."""

    def __init__(self, d: int):
        super().__init__()
        self.proj = nn.Linear(d, d)

    def forward(self, h, h_src):  # pragma: no cover - must never be called
        raise AssertionError("dense V_phi path should not run in this toy")

    def forward_gathered(self, h):
        return torch.tanh(self.proj(h))


def _im(model):
    return scaf.InterventableModel(model, dtype=torch.float64)


# ---------------------------------------------------------------------------
# The loud failure
# ---------------------------------------------------------------------------
def test_unreached_point_raises_instead_of_reporting_a_null_effect():
    with _im(_GatedFockToy(CFG)) as im:
        assert "creation_gates" in im.intervention_names
        with pytest.raises(InertIntervention, match="never fired"):
            with im.do(creation_gates=zero_out):
                im.batch_logits(torch.zeros(1, 8, dtype=torch.long))


def test_no_forward_means_no_complaint():
    """The check must not fire when the block simply never ran a forward.

    Entering ``do()`` to set up state and leaving without a forward pass is
    legitimate; only a *measurement* taken through an inert intervention is a
    false all-clear.
    """
    with _im(_GatedFockToy(CFG)) as im:
        with im.do(creation_gates=zero_out):
            pass


def test_an_error_inside_the_block_propagates_unchanged():
    """A failure in the body must not be masked by the inertness report.

    The body may have raised *before* reaching a forward pass, so complaining
    that the intervention never fired would replace the real cause with a
    confusing symptom.
    """
    with _im(_GatedFockToy(CFG)) as im:
        with pytest.raises(ValueError, match="from the body"):
            with im.do(creation_gates=zero_out):
                raise ValueError("from the body")


# ---------------------------------------------------------------------------
# The points that now work
# ---------------------------------------------------------------------------
def test_modulelist_knockout_reaches_the_children():
    """``destruction_gates`` is indexed, never called: hook the children."""
    model = _GatedFockToy(CFG)
    x = torch.randint(0, CFG.vocab_size, (2, 8))
    with _im(model) as im:
        base = im.batch_logits(x)
        with im.do(destruction_gates=zero_out):
            knocked = im.batch_logits(x)
    assert not torch.allclose(base, knocked), (
        "zeroing every destruction gate must change the output"
    )


def test_bypass_method_knockout_reaches_the_real_entry_point():
    """``V_phi`` is called as ``forward_gathered``, never as ``__call__``."""
    model = _GatedFockToy(CFG)
    x = torch.randint(0, CFG.vocab_size, (2, 8))
    with _im(model) as im:
        base = im.batch_logits(x)
        with im.do(V_phi=zero_out):
            knocked = im.batch_logits(x)
    assert not torch.allclose(base, knocked)


def test_patched_methods_are_restored_on_close():
    model = _GatedFockToy(CFG)
    original = type(model.V_phi).forward_gathered
    im = _im(model)
    assert "forward_gathered" in model.V_phi.__dict__, "wrapper not installed"
    im.close()
    assert "forward_gathered" not in model.V_phi.__dict__
    assert model.V_phi.forward_gathered.__func__ is original


# ---------------------------------------------------------------------------
# Integrator identity, thermostat, trajectories
# ---------------------------------------------------------------------------
class _IntegratorToy(FockLikeToyLM):
    """Carries the config fields the BAOAB/CfC integrator family introduces."""

    def __init__(self, integrator="baoab_cfc", langevin_T=0.5):
        super().__init__(CFG)
        self.cfg.integrator = integrator
        self.cfg.vtheta_analytic_force = True
        self.cfg.langevin_T = langevin_T
        self.cfg.langevin_noise_eval = True

    def forward(self, x):
        logits, _ = super().forward(x)
        if self.cfg.langevin_noise_eval and self.cfg.langevin_T:
            logits = logits + self.cfg.langevin_T * torch.randn_like(logits)
        return logits, None


def test_capabilities_record_which_dynamical_system_was_audited():
    """A verdict that omits the integrator cannot say what it certified."""
    with _im(_IntegratorToy()) as im:
        flags = im.caps.causal_flags
        assert flags["integrator"] == "baoab_cfc"
        assert flags["vtheta_analytic_force"] is True
        assert flags["langevin_T"] == 0.5
        assert any("baoab_cfc" in n for n in im.caps.notes)
        assert any("thermostat" in n for n in im.caps.notes)


def test_deterministic_silences_the_thermostat():
    """Thermal noise inside the forward would read as leakage."""
    model = _IntegratorToy()
    x = torch.randint(0, CFG.vocab_size, (2, 8))
    with _im(model) as im:
        with im.deterministic():
            assert torch.equal(im.batch_logits(x), im.batch_logits(x))
        assert model.cfg.langevin_noise_eval is True, "flag must be restored"


class _InternalApiToy(FockLikeToyLM):
    """Hides the explicit trajectory API so the internal path is exercised.

    The real Fock models have no ``forward_with_trajectory``; the adapter has
    to reconstruct one from ``_embed`` plus ``_stack_forward``, and that is the
    path these tests are about.
    """

    @property
    def forward_with_trajectory(self):
        raise AttributeError("hidden: exercise the _stack_forward path")

    def _embed(self, x):
        return self.emb(x)


class _StackForwardToy(_InternalApiToy):
    """Exposes the real internal trajectory API: ``_stack_forward(h0, x, ...)``.

    The tokens are a genuine second argument — the real stack needs them for
    the per-token mass and the register lifecycle — so an adapter that calls
    ``_stack_forward(h)`` raises ``TypeError`` rather than returning states.
    """

    def _stack_forward(self, h0, x, return_trajectory=False):
        assert x.shape == h0.shape[:2]
        h = self._mix(h0)
        return (h, [h0, h]) if return_trajectory else (h, None)

    def compute_logits(self, h):
        return self.out(h)


def test_trajectory_extraction_passes_the_tokens():
    model = _StackForwardToy(CFG)
    x = torch.randint(0, CFG.vocab_size, (2, 8))
    with _im(model) as im:
        assert im.caps.has_hidden_states
        logits, traj = im.adapter.forward_with_trajectory(model, x)
    assert len(traj) == 2
    assert all(h.shape == (2, 8, CFG.d) for h in traj)
    assert logits.shape == (2, 8, CFG.vocab_size)


def test_hidden_states_are_not_claimed_when_no_trajectory_can_be_returned():
    """``has_hidden_states=True`` on a model that cannot produce them is worse
    than False: the probe runs and dies instead of skipping loudly."""

    class _NoTrajectory(_InternalApiToy):
        def _stack_forward(self, h0, x):
            return self._mix(h0), None

    with _im(_NoTrajectory(CFG)) as im:
        assert not im.caps.has_hidden_states


def test_mediation_records_an_unreached_mediator_instead_of_exonerating_it():
    """An inert mediator must be named, not scored as carrying nothing."""
    corpus = scaf.SyntheticCorpus(CFG.vocab_size, seq_len=32)
    probe = scaf.MediationProbe(
        mediators=("reverse_channel_scale", "creation_gates"),
        n_seqs=4, n_pairs=4, micro_batch=0,
    )
    with _im(_GatedFockToy(CFG)) as im:
        r = probe.run(im, corpus)

    assert r.detail["never_fired"] == ["creation_gates"]
    assert "attributed_creation_gates" not in r.detail, (
        "an unreached mediator must not be given an attribution score"
    )
    assert r.detail["top_mediator"] == "reverse_channel_scale"
