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
            # The reverse channel is the sole carrier of the known leak, so it
            # is the first mediator we knock out.
            mediators.append("reverse_channel_scale")
            if getattr(model, "reverse_ch", None) is not None:
                mediators.append("reverse_ch")
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

        return Capabilities(
            requires_grad_forward=True,
            supports_float64=True,
            has_registers=True,
            has_reverse_channel=has_reverse,
            has_attention=getattr(model, "attn_blocks", None) is not None,
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
            "reverse_ch", "creation_gate", "creation_gate_qkv",
            "creation_gates", "destruction_gates", "V_theta", "V_phi",
            "xi_module", "score_head",
        ):
            mod = getattr(model, attr, None)
            if isinstance(mod, nn.Module):
                points.append((attr, mod))
        return points

    def parameter_interventions(
        self, model: nn.Module
    ) -> Iterable[tuple[str, nn.Parameter]]:
        params: list[tuple[str, nn.Parameter]] = []
        scale = getattr(model, "reverse_channel_scale", None)
        if isinstance(scale, (nn.Parameter, torch.Tensor)):
            params.append(("reverse_channel_scale", scale))
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
