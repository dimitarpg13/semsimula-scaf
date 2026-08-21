"""The adapter contract — SCAF's answer to "what interface must models share?".

SCAF deliberately does **not** require SemSimula models to inherit from a base
class or implement a protocol. Three reasons:

1. The research repo is not an installable package, so SCAF cannot import the
   model classes and therefore cannot ``isinstance``-check them.
2. Forward signatures already diverge across the family (``position_offset``,
   ``kv_caches``, ``return_xi_trajectory``) and return arity varies from 2 to 4.
3. An adapter is *epistemically* better than a base class. ``declared_sources``
   is a human declaration of the model's **intended** causal structure
   (``G_spec``). If the model supplied that itself, it would be marking its own
   homework — which is precisely the failure the Fock audit exposed.

Instead every family gets an adapter that is discovered by **structural (duck)
typing**: it inspects attribute names and class names, never types. Models opt
in to a better adapter only if they want to, by defining ``__scaf_adapter__``.
"""

from __future__ import annotations

import contextlib
from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn

__all__ = ["Capabilities", "ModelAdapter", "PositionalNode"]


# A node in the positional SCM: ("x", t) for a token, ("logit", t) for an
# output, (layer, t) for an internal activation.
PositionalNode = tuple[Any, int]


@dataclass(frozen=True)
class Capabilities:
    """What an adapter knows about the model it wraps.

    Probes consult this to decide which tests apply and to skip (rather than
    silently pass) tests that cannot be run. Skipping loudly matters: a probe
    that quietly returns zero because it never ran is exactly the "false
    all-clear" that axiom A7 exists to prevent.
    """

    #: Conservative-force models compute F = -grad V with ``torch.autograd.grad``
    #: *inside* the forward pass, so the forward cannot run under
    #: ``torch.no_grad()``. This is a family-wide property of SemSimula that
    #: every caller has historically had to rediscover the hard way.
    requires_grad_forward: bool = True

    #: Whether the model can be cast to float64 for a bit-exact determinism
    #: baseline (axiom A6).
    supports_float64: bool = True

    #: Structural features that gate specific probes.
    has_registers: bool = False
    has_reverse_channel: bool = False
    has_attention: bool = False

    #: Whether the adapter can return per-layer hidden-state trajectories.
    #: Required for Tier A (hidden-state cosine deviation) probes.
    has_hidden_states: bool = False

    #: Whether the adapter can return Gaussian well parameters (centres,
    #: precision, weights) for each layer. Required for Tier B
    #: (basin-membership) probes.
    has_vtheta_wells: bool = False

    #: Named intervention points usable as mediators in a knockout (axiom A5),
    #: most-suspect first.
    mediators: tuple[str, ...] = ()

    #: Causality-relevant config flags, verbatim from the model config. These
    #: are reported in the scorecard so an audit record states which causal
    #: guarantees the model *claimed* at audit time.
    causal_flags: Mapping[str, Any] = field(default_factory=dict)

    #: Free-form notes surfaced in the scorecard.
    notes: tuple[str, ...] = ()

    def describe(self) -> str:
        bits = []
        if self.has_registers:
            bits.append("registers")
        if self.has_reverse_channel:
            bits.append("reverse-channel")
        if self.has_attention:
            bits.append("attention")
        if self.has_hidden_states:
            bits.append("hidden-states")
        if self.has_vtheta_wells:
            bits.append("vtheta-wells")
        return ", ".join(bits) if bits else "plain"


class ModelAdapter(ABC):
    """Family-specific plug telling SCAF how to drive and probe a model.

    Subclasses must be registered with
    :func:`scaf.core.adapters.register_adapter` so auto-detection can find them.
    """

    #: Human-readable adapter name, recorded in the scorecard.
    name: str = "abstract"

    #: Detection priority; higher wins when several adapters match. The generic
    #: fallback sits at 0, specific families above it.
    priority: int = 0

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------
    @classmethod
    @abstractmethod
    def detect(cls, model: nn.Module) -> bool:
        """Return True if this adapter can drive ``model``.

        Implementations must use structural checks (``hasattr``, class-name
        strings) and must never import or ``isinstance``-check SemSimula types.
        """

    # ------------------------------------------------------------------
    # Model introspection
    # ------------------------------------------------------------------
    @abstractmethod
    def capabilities(self, model: nn.Module) -> Capabilities:
        """Describe what this model supports and which mediators it exposes."""

    def config(self, model: nn.Module) -> dict[str, Any]:
        """Best-effort extraction of ``d``, ``L``, ``vocab_size``, ``max_len``.

        The default reads a ``cfg`` attribute, which every SemSimula model has.
        """
        cfg = getattr(model, "cfg", None)
        if cfg is None:
            return {}
        keys = (
            "vocab_size", "d", "L", "max_len", "causal_force",
            "prefix_causal_registers", "reverse_channel", "n_registers",
            "xi_channels", "top_k", "stack_discipline", "fock_version",
            # Integrator identity: 'verlet' vs the BAOAB/CfC family. Two runs
            # of the same architecture under different integrators are
            # different dynamical systems, so an audit record that omits this
            # cannot say which one it certified.
            "integrator", "vtheta_analytic_force", "langevin_T",
        )
        return {k: getattr(cfg, k) for k in keys if hasattr(cfg, k)}

    # ------------------------------------------------------------------
    # Driving the model
    # ------------------------------------------------------------------
    def forward_logits(self, model: nn.Module, x: torch.Tensor) -> torch.Tensor:
        """Run a forward pass and return detached logits of shape (B, T, V).

        Handles the two things that trip up every naive caller:

        * **Grad must stay enabled.** Conservative-force models call
          ``torch.autograd.grad`` inside the forward pass, so wrapping this in
          ``torch.no_grad()`` raises. We force grad on regardless of the
          caller's ambient context.
        * **Return arity varies.** Depending on flags a model returns
          ``(logits, loss)``, ``(logits, loss, traj)`` or
          ``(logits, loss, traj, caches)``. We always take element 0.

        Logits are detached immediately so the autograd graph is freed rather
        than accumulating across probe iterations.
        """
        with torch.enable_grad():
            out = model(x)
        logits = out[0] if isinstance(out, (tuple, list)) else out
        return logits.detach()

    def forward_with_trajectory(
        self, model: nn.Module, x: torch.Tensor
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Run a forward pass and return ``(logits, [h_0, h_1, ..., h_L])``.

        Each ``h_ℓ`` has shape ``(B, T, d)`` — the hidden state at position
        ``t`` after layer ``ℓ`` has executed. ``h_0`` is the embedding layer
        output. ``h_L`` is the final hidden state before the output projection.

        Required for Tier A geometric probes (hidden-state cosine deviation).
        Adapters that do not support trajectories leave
        ``Capabilities.has_hidden_states = False`` and probes skip loudly.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support forward_with_trajectory. "
            "Set has_hidden_states=True in capabilities and implement this "
            "method to enable geometric probes."
        )

    def well_parameters(
        self, model: nn.Module, layer_idx: int, x: torch.Tensor
    ) -> dict[str, torch.Tensor] | None:
        """Return Gaussian well parameters for a given layer and input batch.

        For models with anisotropic Gaussian :math:`V_\\theta`, the well
        parameters are **context-dependent** — they are functions of the
        learned context vectors ``xi``, which in turn depend on the input
        tokens. This is why the input batch ``x`` is required: the adapter
        re-derives ``xi`` from ``x`` to extract the well parameters that were
        active during the original forward pass.

        Returns a dict with detached tensors::

            {
                'mu':             (B, K, d)     — well centres,
                'precision_diag': (B, K, d)     — diagonal precision ``a_k``,
                'precision_lr':   (B, K, d, r)  — low-rank factor ``B_k``,
                'weights':        (B, K)        — mixture weights,
            }

        or ``None`` if the model does not have Gaussian wells.

        The full precision matrix is :math:`\\Sigma_k^{-1} = \\mathrm{diag}(a_k) + B_k B_k^\\top`,
        but it is never materialised — the quadratic form is evaluated via the
        diagonal + low-rank decomposition for :math:`O(d \\cdot r)` cost.

        Required for Tier B (basin-membership) probes. Adapters that do not
        support wells leave ``Capabilities.has_vtheta_wells = False`` and
        probes skip loudly.
        """
        return None

    @contextlib.contextmanager
    def deterministic(self, model: nn.Module) -> Iterator[None]:
        """Best-effort context that removes stochasticity from the forward pass.

        This is only an aid — :class:`~scaf.probes.controls.DeterminismControl`
        independently verifies that repeated forwards agree bit-for-bit, so a
        model this fails to tame is caught rather than trusted.
        """
        was_training = model.training
        model.eval()
        try:
            yield
        finally:
            model.train(was_training)

    # ------------------------------------------------------------------
    # Intervention surface
    # ------------------------------------------------------------------
    def intervention_points(
        self, model: nn.Module
    ) -> Iterable[tuple[str, nn.Module]]:
        """Yield ``(name, module)`` pairs that SCAF may hook for do-operations.

        Names become the keyword arguments accepted by
        :meth:`scaf.core.intervenable.InterventableModel.do`.
        """
        return ()

    def intervention_methods(
        self, model: nn.Module
    ) -> Iterable[tuple[str, nn.Module, str]]:
        """Yield ``(name, module, method_name)`` for entry points that bypass ``__call__``.

        A forward hook only fires when a module is invoked through
        ``__call__``. Several SemSimula modules are instead invoked through a
        named method — ``V_phi.forward_gathered``, ``creation_gate_qkv.
        forward_prefix``, ``V_theta.analytical_grad`` — chosen by a config
        flag. A hook registered on such a module never fires, so a knockout
        of it would be a silent no-op, which is exactly the false all-clear
        axiom A7 exists to prevent.

        Declaring the method here makes SCAF wrap it, so the edit applies to
        whatever the module actually produces. Adapters should declare only
        the methods that apply *in the model's current configuration*; if a
        method and ``__call__`` both run during one pass, the edit is applied
        to both, which is the intended semantics for a knockout.
        """
        return ()

    def parameter_interventions(
        self, model: nn.Module
    ) -> Iterable[tuple[str, nn.Parameter]]:
        """Yield ``(name, parameter)`` pairs that may be clamped directly.

        Used for scalar gates (e.g. a reverse-channel scale) where clamping the
        parameter is a cleaner knockout than hooking a module output.
        """
        return ()

    # ------------------------------------------------------------------
    # Declared causal structure (G_spec)
    # ------------------------------------------------------------------
    def declared_sources(
        self, layer: int, t: int, T: int
    ) -> Iterable[PositionalNode]:
        """The nodes that node ``(layer, t)`` is *intended* to depend on.

        This is the adapter author's honest statement of intent. SCAF diffs it
        against measured behaviour; any edge from a position ``> t`` is a
        candidate leak. The default is the strict autoregressive triangle.
        """
        if layer == 0:
            return [("x", s) for s in range(t + 1)]
        return [(layer - 1, s) for s in range(t + 1)]

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<{type(self).__name__} name={self.name!r}>"
