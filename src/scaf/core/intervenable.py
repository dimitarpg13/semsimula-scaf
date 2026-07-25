"""``InterventableModel`` — turn a PyTorch model into a manipulable SCM.

A SemSimula forward pass *is* a structural causal model: tokens and parameters
are exogenous, every activation is endogenous and assigned by a deterministic
structural equation (the layer step). Unlike the observational settings DoWhy
targets, we own the data-generating process, so identification is free — we can
simply set any node and re-run.

This class is the bridge. It exposes do-operations on

* **input tokens** — via the ``tokens=`` argument to :meth:`logits`,
* **named internal tensors** — via :meth:`do`, using forward hooks on the
  modules the adapter declared as intervention points,
* **named parameters** — via :meth:`clamp`, for scalar gates where clamping the
  parameter is a cleaner knockout than hooking a module output.

Interventions are scoped by context managers, so a single loaded checkpoint
serves the entire probe battery without reloading.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Iterator, Sequence

import torch
from torch import nn

from .adapters import ModelAdapter, resolve_adapter

__all__ = ["InterventableModel"]

TensorEdit = Callable[[torch.Tensor], torch.Tensor]


def zero_out(t: torch.Tensor) -> torch.Tensor:
    """Edit function that knocks a tensor out entirely."""
    return torch.zeros_like(t)


class InterventableModel:
    """Wrap a model as a manipulable SCM.

    Args:
        model: The model under test. Used in-place; see ``dtype`` below.
        adapter: Optional explicit adapter. Auto-detected when omitted.
        device: Device to run on.
        dtype: Cast the model to this dtype. ``torch.float64`` gives the
            bit-exact determinism baseline required by axiom A6 and is the
            right choice for architectural probes on small models. Use
            ``torch.float32`` for trained-scale probes on real checkpoints,
            where float64 would double an already-large memory footprint.
            ``None`` leaves the model untouched.
        restore_dtype: Whether to restore the original dtype on :meth:`close`.
    """

    def __init__(
        self,
        model: nn.Module,
        adapter: ModelAdapter | None = None,
        device: str | torch.device = "cpu",
        dtype: torch.dtype | None = None,
        restore_dtype: bool = True,
    ) -> None:
        self.adapter = resolve_adapter(model, adapter)
        self.device = torch.device(device)
        self.model = model
        self.caps = self.adapter.capabilities(model)

        self._orig_dtype = next(
            (p.dtype for p in model.parameters()), torch.float32
        )
        self._restore_dtype = restore_dtype
        if dtype is not None:
            if dtype == torch.float64 and not self.caps.supports_float64:
                raise ValueError(
                    f"{self.adapter.name} adapter reports the model does not "
                    "support float64"
                )
            self.model = self.model.to(dtype=dtype)
        self.model = self.model.to(device=self.device)
        self.dtype = dtype or self._orig_dtype

        self._edits: dict[str, TensorEdit] = {}
        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        self._points: dict[str, nn.Module] = {}
        self._params: dict[str, nn.Parameter] = dict(
            self.adapter.parameter_interventions(self.model)
        )

        for name, module in self.adapter.intervention_points(self.model):
            self._points[name] = module
            self._handles.append(
                module.register_forward_hook(self._make_hook(name))
            )

        self.n_forwards = 0

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------
    @property
    def intervention_names(self) -> tuple[str, ...]:
        """Names accepted by :meth:`do`."""
        return tuple(self._points)

    @property
    def clampable_names(self) -> tuple[str, ...]:
        """Names accepted by :meth:`clamp`."""
        return tuple(self._params)

    @property
    def mediators(self) -> tuple[str, ...]:
        """Mediator names the adapter considers worth knocking out."""
        return self.caps.mediators

    def config(self) -> dict:
        return self.adapter.config(self.model)

    # ------------------------------------------------------------------
    # Do-operations
    # ------------------------------------------------------------------
    def _make_hook(self, name: str):
        def hook(_module, _inputs, output):
            fn = self._edits.get(name)
            if fn is None:
                return output
            if isinstance(output, tuple):
                # Edit only the primary tensor of a tuple-returning module.
                return (fn(output[0]), *output[1:])
            return fn(output)

        return hook

    @contextlib.contextmanager
    def do(self, **edits: TensorEdit) -> Iterator[InterventableModel]:
        """Scope one or more do-operations on named internal tensors.

        Example::

            with im.do(reverse_ch=zero_out):
                leaked = probe.run(im, corpus)

        Raises:
            KeyError: if a name is not an intervention point. Failing loudly
                matters here: a silently-ignored knockout would make a leaky
                model look clean.
        """
        unknown = set(edits) - set(self._points)
        if unknown:
            raise KeyError(
                f"Unknown intervention point(s) {sorted(unknown)}. "
                f"Available: {sorted(self._points)}"
            )
        prev = self._edits
        self._edits = {**self._edits, **edits}
        try:
            yield self
        finally:
            self._edits = prev

    @contextlib.contextmanager
    def clamp(self, **values: float) -> Iterator[InterventableModel]:
        """Scope a clamp of named parameters to constant values.

        Example::

            with im.clamp(reverse_channel_scale=0.0):
                ...
        """
        unknown = set(values) - set(self._params)
        if unknown:
            raise KeyError(
                f"Unknown clampable parameter(s) {sorted(unknown)}. "
                f"Available: {sorted(self._params)}"
            )
        saved = {k: self._params[k].detach().clone() for k in values}
        try:
            with torch.no_grad():
                for k, v in values.items():
                    self._params[k].fill_(v)
            yield self
        finally:
            with torch.no_grad():
                for k, original in saved.items():
                    self._params[k].copy_(original)

    @contextlib.contextmanager
    def knockout(self, mediator: str) -> Iterator[InterventableModel]:
        """Knock out a mediator, whichever mechanism applies.

        Prefers clamping a parameter to zero; falls back to zeroing a module's
        output. This lets :class:`~scaf.probes.mediation.MediationProbe` name a
        mediator without caring how it is realised.
        """
        if mediator in self._params:
            with self.clamp(**{mediator: 0.0}):
                yield self
        elif mediator in self._points:
            with self.do(**{mediator: zero_out}):
                yield self
        else:
            raise KeyError(
                f"'{mediator}' is neither a clampable parameter nor an "
                f"intervention point. Clampable: {sorted(self._params)}; "
                f"points: {sorted(self._points)}"
            )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def logits(self, tokens: Sequence[int] | torch.Tensor) -> torch.Tensor:
        """Return logits for a single sequence, shape ``(T, V)``.

        Accepts a 1-D sequence (treated as one batch element) or a 2-D batch,
        in which case the full ``(B, T, V)`` tensor is returned.
        """
        x = self._as_tokens(tokens)
        self.n_forwards += 1
        out = self.adapter.forward_logits(self.model, x)
        return out[0] if out.dim() == 3 and x.shape[0] == 1 else out

    def batch_logits(self, tokens: Sequence[Sequence[int]] | torch.Tensor) -> torch.Tensor:
        """Return logits for a batch, always shape ``(B, T, V)``."""
        x = self._as_tokens(tokens)
        self.n_forwards += 1
        return self.adapter.forward_logits(self.model, x)

    def _as_tokens(self, tokens) -> torch.Tensor:
        # Note the deliberate absence of torch.from_numpy: it fails when torch
        # and NumPy disagree on major version. as_tensor with an explicit dtype
        # goes through the buffer protocol and survives that mismatch.
        if isinstance(tokens, torch.Tensor):
            x = tokens.to(device=self.device, dtype=torch.long)
        else:
            x = torch.as_tensor(tokens, dtype=torch.long).to(self.device)
        if x.dim() == 1:
            x = x[None]
        elif x.dim() != 2:
            raise ValueError(f"expected 1-D or 2-D token tensor, got {x.shape}")
        return x

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def deterministic(self):
        """Adapter-specific context that suppresses stochastic routing."""
        return self.adapter.deterministic(self.model)

    def close(self) -> None:
        """Remove hooks and optionally restore the original dtype."""
        for h in self._handles:
            h.remove()
        self._handles.clear()
        if self._restore_dtype and self.dtype != self._orig_dtype:
            self.model = self.model.to(dtype=self._orig_dtype)
            self.dtype = self._orig_dtype

    def __enter__(self) -> InterventableModel:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"<InterventableModel adapter={self.adapter.name!r} "
            f"dtype={self.dtype} device={self.device} "
            f"points={len(self._points)} params={len(self._params)}>"
        )
