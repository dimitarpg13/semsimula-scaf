"""Adapter for the Fock register families (``FockPARFLM_v2``, ``FockMultiXiPARFLM``).

This is the family where the known leak lived, so it is the adapter with the
richest intervention surface. The leak pathway, for reference:

    future token  ->  creation gate  ->  shared register state
                  ->  reverse channel  ->  past token state  ->  past logit

Before the prefix-causal fix the register state was a single ``(B, M, d)``
tensor shared by every position, so a register written by token ``s`` was
readable at any position ``t < s``. The fix expands it to ``(B, T, M, d)`` so
position ``t`` only ever reads registers written by tokens ``<= t``.

Detection is purely structural — no SemSimula import — so this file works
whether or not the research repo is on ``sys.path``.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterable, Iterator

import torch
from torch import nn

from .base import Capabilities, ModelAdapter

__all__ = ["FockAdapter"]


#: Attribute names that mark a Fock-style register lifecycle. Names differ
#: slightly between ``FockPARFLM_v2`` and ``FockMultiXiPARFLM``.
_REGISTER_MARKERS = ("register_embed",)
_REVERSE_MARKERS = ("reverse_ch", "reverse_channel_scale")
_CREATION_MARKERS = ("creation_gate", "creation_gate_qkv", "creation_gates")
_DESTRUCTION_MARKERS = ("destruction_gates", "destruction_gate")

#: ``FockAttentionPARFLM`` adds a direct exchange force behind its own
#: ``torch.tanh(exchange_scale)`` gate — structurally the same kind of
#: non-conservative channel as the reverse channel, and therefore just as
#: capable of carrying a leak. It must be a first-class mediator: omitting it
#: would let mediation report a confident attribution while never testing one
#: of the two candidate carriers.
_EXCHANGE_MARKERS = ("exchange_force", "exchange_scale")

#: Learnable gates whose zero value closes the channel, because they are read
#: through ``torch.tanh``. Clamping any of these to zero is a valid knockout.
#: ``logit_scale`` is deliberately excluded: it is an output temperature, not a
#: routing gate, and zeroing it would distort every logit rather than isolate
#: a path.
_CLAMPABLE_GATES = ("reverse_channel_scale", "exchange_scale")


def _has_gaussian_wells(model: nn.Module) -> bool:
    """Check whether the model's V_theta exposes Gaussian well parameters.

    True for models using ``AnisotropicDepthConditionedGaussianVTheta`` or
    ``AnisotropicMultiContextGaussianVTheta`` — detected structurally via
    ``_components`` and ``mu_proj``.  Also true for toy models that expose
    ``well_parameters`` directly.
    """
    vtheta = getattr(model, "V_theta", None)
    if vtheta is None:
        return hasattr(model, "well_parameters")
    # AnisotropicDepthConditioned wraps a bank; the bank wraps per-head banks.
    bank = getattr(vtheta, "bank", vtheta)
    if hasattr(bank, "banks"):
        # AnisotropicMultiContextGaussianVTheta — check the first head
        heads = bank.banks
        if hasattr(heads, "__getitem__") and len(heads) > 0:
            bank = heads[0]
    return hasattr(bank, "_components") and hasattr(bank, "mu_proj")


def _first_attr(model: nn.Module, names: Iterable[str]) -> str | None:
    for n in names:
        if getattr(model, n, None) is not None:
            return n
    return None


class FockAdapter(ModelAdapter):
    """Drives Fock v2 / Fock-MultiXi models and exposes their leak channels."""

    name = "fock"
    priority = 100

    @classmethod
    def detect(cls, model: nn.Module) -> bool:
        has_registers = _first_attr(model, _REGISTER_MARKERS) is not None
        has_creation = _first_attr(model, _CREATION_MARKERS) is not None
        if has_registers and has_creation:
            return True
        # Fall back to the class name for partially-constructed models.
        return "fock" in type(model).__name__.lower() and has_registers

    # ------------------------------------------------------------------
    def capabilities(self, model: nn.Module) -> Capabilities:
        cfg = getattr(model, "cfg", None)
        rev_attr = _first_attr(model, _REVERSE_MARKERS)
        has_reverse = rev_attr is not None

        mediators: list[str] = []
        if has_reverse:
            # The reverse channel carried the known leak, so it is knocked out
            # first — but being first is a prior, not a conclusion.
            if getattr(model, "reverse_channel_scale", None) is not None:
                mediators.append("reverse_channel_scale")
            if getattr(model, "reverse_ch", None) is not None:
                mediators.append("reverse_ch")
        exchange = _first_attr(model, _EXCHANGE_MARKERS)
        if exchange:
            mediators.append(exchange)
        creation = _first_attr(model, _CREATION_MARKERS)
        if creation:
            mediators.append(creation)
        destruction = _first_attr(model, _DESTRUCTION_MARKERS)
        if destruction:
            mediators.append(destruction)

        flags = {}
        notes: list[str] = []
        if cfg is not None:
            for key in (
                "causal_force", "prefix_causal_registers", "reverse_channel",
                "n_registers", "stack_discipline", "fock_version",
                "xi_channels", "top_k",
            ):
                if hasattr(cfg, key):
                    flags[key] = getattr(cfg, key)

            if hasattr(cfg, "prefix_causal_registers"):
                if not cfg.prefix_causal_registers:
                    notes.append(
                        "prefix_causal_registers=False: this is the PRE-FIX "
                        "leaky register lifecycle. A large tau_leak is expected."
                    )
            else:
                notes.append(
                    "Model config has no 'prefix_causal_registers' field, so it "
                    "predates the causal-leak fix entirely."
                )
            if getattr(cfg, "gumbel_noise", False):
                notes.append(
                    "gumbel_noise=True: stochastic routing may break the "
                    "bit-exact determinism baseline; DeterminismControl will "
                    "report it."
                )

        has_trajectory = (
            hasattr(model, "_stack_forward")
            or hasattr(model, "forward_with_trajectory")
        )
        has_wells = _has_gaussian_wells(model)

        return Capabilities(
            requires_grad_forward=True,
            supports_float64=True,
            has_registers=True,
            has_reverse_channel=has_reverse,
            has_attention=getattr(model, "attn_blocks", None) is not None,
            has_hidden_states=has_trajectory,
            has_vtheta_wells=has_wells,
            mediators=tuple(mediators),
            causal_flags=flags,
            notes=tuple(notes),
        )

    # ------------------------------------------------------------------
    @contextlib.contextmanager
    def deterministic(self, model: nn.Module) -> Iterator[None]:
        """Freeze routing stochasticity on top of the base eval-mode switch."""
        was_training = model.training
        cfg = getattr(model, "cfg", None)
        prev_noise = getattr(cfg, "gumbel_noise", None) if cfg else None
        model.eval()
        if cfg is not None and prev_noise is not None:
            cfg.gumbel_noise = False
        try:
            yield
        finally:
            if cfg is not None and prev_noise is not None:
                cfg.gumbel_noise = prev_noise
            model.train(was_training)

    # ------------------------------------------------------------------
    def intervention_points(
        self, model: nn.Module
    ) -> Iterable[tuple[str, nn.Module]]:
        points: list[tuple[str, nn.Module]] = []
        for attr in (
            "reverse_ch", "exchange_force", "creation_gate",
            "creation_gate_qkv", "creation_gates", "destruction_gates",
            "V_theta", "V_phi", "xi_module", "score_head",
        ):
            mod = getattr(model, attr, None)
            if isinstance(mod, nn.Module):
                points.append((attr, mod))
        return points

    def parameter_interventions(
        self, model: nn.Module
    ) -> Iterable[tuple[str, nn.Parameter]]:
        params: list[tuple[str, nn.Parameter]] = []
        for name in _CLAMPABLE_GATES:
            gate = getattr(model, name, None)
            if isinstance(gate, (nn.Parameter, torch.Tensor)):
                params.append((name, gate))
        return params

    # ------------------------------------------------------------------
    def open_reverse_channel(self, model: nn.Module, value: float = 5.0) -> bool:
        """Force the reverse-channel gate wide open.

        Used by the architectural probe: a *structurally* causal model must
        show exactly zero future influence even with the leak's carrier
        maximally amplified. Returns True if a gate was found and set.
        """
        found = False
        with torch.no_grad():
            for name, param in model.named_parameters():
                if "reverse_channel_scale" in name:
                    param.fill_(value)
                    found = True
        return found

    # ------------------------------------------------------------------
    def well_parameters(
        self, model: nn.Module, layer_idx: int, x: torch.Tensor
    ) -> dict[str, torch.Tensor] | None:
        """Extract Gaussian well parameters for a given layer.

        Handles three model shapes:

        1. ``model.well_parameters(layer_idx, x)`` — explicit API (toy models).
        2. ``AnisotropicDepthConditionedGaussianVTheta`` — sets active layer,
           derives ``xi`` from ``x``, calls ``bank._components``.
        3. Returns ``None`` if the model has no Gaussian wells.
        """
        if hasattr(model, "well_parameters"):
            return model.well_parameters(layer_idx, x)

        vtheta = getattr(model, "V_theta", None)
        if vtheta is None:
            return None

        xi_mod = getattr(model, "xi_module", None)
        if xi_mod is None:
            return None

        with torch.no_grad():
            xi = xi_mod(x)

        # DepthConditioned: set the layer and apply the depth-code shift
        if hasattr(vtheta, "set_active_layer") and hasattr(vtheta, "_shift"):
            vtheta.set_active_layer(layer_idx)
            xi_shifted = vtheta._shift(xi)  # noqa: SLF001
        else:
            xi_shifted = xi

        # Reach the actual Gaussian bank that has _components
        bank = getattr(vtheta, "bank", vtheta)
        if hasattr(bank, "banks"):
            # MultiContext: extract from each head and concatenate
            heads = bank.banks
            all_mu, all_a, all_w, all_B = [], [], [], []
            for m_idx, head in enumerate(heads):
                xi_head = xi_shifted[..., m_idx, :]
                mu, a, w, B = head._components(xi_head)  # noqa: SLF001
                all_mu.append(mu)
                all_a.append(a)
                all_w.append(w)
                all_B.append(B)
            mu = torch.cat(all_mu, dim=-2)
            a = torch.cat(all_a, dim=-2)
            w = torch.cat(all_w, dim=-1)
            B = torch.cat(all_B, dim=-3)
        elif hasattr(bank, "_components"):
            mu, a, w, B = bank._components(xi_shifted)  # noqa: SLF001
        else:
            return None

        return {
            "mu": mu.detach(),
            "precision_diag": a.detach(),
            "precision_lr": B.detach(),
            "weights": w.detach(),
        }

    # ------------------------------------------------------------------
    def forward_with_trajectory(
        self, model: nn.Module, x: torch.Tensor
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Return ``(logits, [h_0, ..., h_L])`` using the model's own trajectory support.

        Three paths, in priority order:

        1. ``model.forward_with_trajectory(x)`` — explicit trajectory API
           (used by toy models in the test suite).
        2. ``model._embed(x)`` + ``model._stack_forward(h, ..., return_trajectory=True)``
           — the real Fock-PARFLM internal API.
        3. Raises ``NotImplementedError`` if neither is available.

        Logits and hidden states are detached so the autograd graph is freed
        after each pass, matching the contract of ``forward_logits``.
        """
        with torch.enable_grad():
            if hasattr(model, "forward_with_trajectory"):
                out = model.forward_with_trajectory(x)
                logits = out[0].detach()
                trajectory = [h.detach() for h in out[1]]
                return logits, trajectory

            if hasattr(model, "_stack_forward") and hasattr(model, "_embed"):
                h = model._embed(x)
                sf_out = model._stack_forward(h, return_trajectory=True)
                # _stack_forward returns (h_final, trajectory) or
                # (h_final, loss, trajectory) depending on the version.
                if isinstance(sf_out[-1], (list, tuple)):
                    traj_raw = sf_out[-1]
                else:
                    traj_raw = [sf_out[0]]
                logits_in = traj_raw[-1] if traj_raw else sf_out[0]
                score_head = getattr(model, "score_head", None)
                if score_head is not None:
                    logits = score_head(logits_in).detach()
                else:
                    logits = model(x)
                    logits = (logits[0] if isinstance(logits, (tuple, list))
                              else logits).detach()
                trajectory = [h.detach() for h in traj_raw]
                return logits, trajectory

        raise NotImplementedError(
            "FockAdapter.forward_with_trajectory: model has neither "
            "'forward_with_trajectory' nor '_embed'+'_stack_forward'. "
            "Cannot extract hidden-state trajectories."
        )
