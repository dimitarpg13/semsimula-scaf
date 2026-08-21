"""Fallback adapter for any autoregressive LM SCAF does not specifically know.

This is what makes SCAF usable beyond the models it ships adapters for. It
assumes only the weakest possible contract:

* ``model(x)`` accepts a ``(B, T)`` long tensor,
* and returns logits ``(B, T, V)`` either directly or as element 0 of a tuple.

Every SemSimula model satisfies this, as does a plain GPT-style decoder. The
token-level probes (future perturbation, target relocation, the controls) need
nothing more, so a model with no dedicated adapter still gets a real audit —
just without component attribution, since the generic adapter cannot know
which submodule is the mediator.
"""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn

from .base import Capabilities, ModelAdapter

__all__ = ["GenericAdapter"]


#: Submodule names that are plausible mediators across many architectures.
#: Purely heuristic — offered so mediation is *possible* on unknown models,
#: while the scorecard makes clear the attribution is best-effort.
_CANDIDATE_MEDIATORS = (
    "reverse_ch",
    "reverse_channel_scale",
    "exchange_force",
    "exchange_scale",
    "nonconservative",
    "skew_kernel",
    "gyro_kernel",
    "V_attn",
    "xi_module",
    "V_phi",
    "V_theta",
)


class GenericAdapter(ModelAdapter):
    """Lowest-priority adapter; matches any ``nn.Module`` that looks like an LM."""

    name = "generic"
    priority = 0

    @classmethod
    def detect(cls, model: nn.Module) -> bool:
        return isinstance(model, nn.Module)

    def capabilities(self, model: nn.Module) -> Capabilities:
        cfg = getattr(model, "cfg", None)
        mediators = tuple(
            n for n in _CANDIDATE_MEDIATORS if getattr(model, n, None) is not None
        )

        flags = {}
        if cfg is not None:
            for key in (
                "causal_force", "prefix_causal_registers", "reverse_channel",
                "n_registers", "xi_channels", "top_k",
            ):
                if hasattr(cfg, key):
                    flags[key] = getattr(cfg, key)

        notes = [
            "Generic adapter: token-level probes are fully valid, but component "
            "attribution is heuristic because the model family is unknown."
        ]

        has_trajectory = (
            hasattr(model, "_stack_forward")
            or hasattr(model, "forward_with_trajectory")
        )
        has_wells = hasattr(model, "well_parameters")

        return Capabilities(
            requires_grad_forward=True,
            supports_float64=True,
            has_registers=getattr(model, "register_embed", None) is not None,
            has_reverse_channel=any(
                getattr(model, n, None) is not None
                for n in ("reverse_ch", "reverse_channel_scale")
            ),
            has_attention=getattr(model, "attn_blocks", None) is not None,
            has_hidden_states=has_trajectory,
            has_vtheta_wells=has_wells,
            mediators=mediators,
            causal_flags=flags,
            notes=tuple(notes),
        )

    def forward_with_trajectory(
        self, model: nn.Module, x: torch.Tensor
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Delegate to the model's own ``forward_with_trajectory`` if it exists."""
        if hasattr(model, "forward_with_trajectory"):
            with torch.enable_grad():
                out = model.forward_with_trajectory(x)
            logits = out[0].detach()
            trajectory = [h.detach() for h in out[1]]
            return logits, trajectory
        return super().forward_with_trajectory(model, x)

    def well_parameters(
        self,
        model: nn.Module,
        layer_idx: int,
        x: torch.Tensor,
        h: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor] | None:
        """Delegate to the model's own ``well_parameters`` if it exists."""
        if hasattr(model, "well_parameters"):
            return model.well_parameters(layer_idx, x)
        return None

    def intervention_points(
        self, model: nn.Module
    ) -> Iterable[tuple[str, nn.Module]]:
        return [
            (n, m)
            for n in _CANDIDATE_MEDIATORS
            if isinstance(m := getattr(model, n, None), nn.Module)
        ]

    def parameter_interventions(self, model: nn.Module):
        out = []
        for name in ("reverse_channel_scale", "exchange_scale"):
            p = getattr(model, name, None)
            if isinstance(p, nn.Parameter):
                out.append((name, p))
        return out
