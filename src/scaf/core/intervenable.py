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

__all__ = ["InertIntervention", "InterventableModel", "resolve_dtype"]


class InertIntervention(RuntimeError):
    """Raised when an intervention was registered but never actually applied.

    A distinct type, rather than a bare ``RuntimeError``, so callers can tell
    "this knockout did nothing because the point is unreachable" from "the
    model raised while running the knockout". The first is a statement about
    the adapter's intervention surface and must never be read as the component
    being innocent; the second is a genuine failure that should propagate.
    """

TensorEdit = Callable[[torch.Tensor], torch.Tensor]

_DTYPES = {
    "float64": torch.float64, "f64": torch.float64, "double": torch.float64,
    "float32": torch.float32, "f32": torch.float32, "float": torch.float32,
}


def resolve_dtype(dtype):
    """Accept ``"float64"`` and friends wherever a ``torch.dtype`` is expected.

    Resolution lives here, at the single point every entry path funnels
    through, rather than in each public function. Duplicating it once produced
    an API where ``audit(dtype="float64")`` worked and
    ``build_leak_frame(dtype="float64")`` raised a ``TypeError`` from deep
    inside ``Module.to``.
    """
    if dtype is None or isinstance(dtype, torch.dtype):
        return dtype
    key = str(dtype).lower().replace("torch.", "")
    if key not in _DTYPES:
        raise ValueError(f"unknown dtype {dtype!r}; use one of {sorted(_DTYPES)}")
    return _DTYPES[key]


def zero_out(t: torch.Tensor) -> torch.Tensor:
    """Edit function that knocks a tensor out entirely."""
    return torch.zeros_like(t)


def _hookable_targets(module: nn.Module) -> list[nn.Module]:
    """Expand pure containers into the children that actually run.

    ``nn.ModuleList`` and ``nn.ModuleDict`` are indexed, never called: their
    ``forward`` raises. A hook registered on the container is therefore dead,
    and a per-layer gate stack (``destruction_gates[l](r)``) would look
    intervenable while being untouchable. Hooking the children instead applies
    the edit at every layer, which is what knocking out "the destruction
    gates" means.
    """
    if isinstance(module, (nn.ModuleList, nn.ModuleDict)):
        targets: list[nn.Module] = []
        for child in module.children():
            targets.extend(_hookable_targets(child))
        return targets
    return [module]


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
        dtype = resolve_dtype(dtype)

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
        self._points: dict[str, list[nn.Module]] = {}
        self._patched: list[tuple[nn.Module, str]] = []
        self._fired: set[str] = set()
        self._params: dict[str, nn.Parameter] = dict(
            self.adapter.parameter_interventions(self.model)
        )

        for name, module in self.adapter.intervention_points(self.model):
            targets = _hookable_targets(module)
            self._points.setdefault(name, []).extend(targets)
            for target in targets:
                self._handles.append(
                    target.register_forward_hook(self._make_hook(name))
                )

        # Entry points that bypass __call__ and would therefore never see a
        # forward hook. See ModelAdapter.intervention_methods.
        for name, module, method_name in self.adapter.intervention_methods(
            self.model
        ):
            self._points.setdefault(name, [])
            self._patch_method(name, module, method_name)

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
    def _apply(self, name: str, output):
        """Record that ``name``'s entry point ran, and apply any pending edit."""
        self._fired.add(name)
        fn = self._edits.get(name)
        if fn is None:
            return output
        if isinstance(output, tuple):
            # Edit only the primary tensor of a tuple-returning module.
            return (fn(output[0]), *output[1:])
        return fn(output)

    def _make_hook(self, name: str):
        def hook(_module, _inputs, output):
            return self._apply(name, output)

        return hook

    def _patch_method(self, name: str, module: nn.Module, method_name: str):
        """Route ``module.method_name`` through the edit machinery.

        The wrapper is installed as an instance attribute, shadowing the bound
        class method, and removed again by :meth:`close`.
        """
        original = getattr(module, method_name)

        def wrapper(*args, **kwargs):
            return self._apply(name, original(*args, **kwargs))

        wrapper.__name__ = getattr(original, "__name__", method_name)
        wrapper.__scaf_wrapped__ = original
        setattr(module, method_name, wrapper)
        self._patched.append((module, method_name))

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
            InertIntervention: on exit, if forwards ran inside the block but
                an edited point never fired. Knowing a point *exists* is not the
                same as knowing it was *reached*: a module invoked through a
                bypass method, or one whose container is never called, accepts
                a hook that no forward pass will ever trigger. The block then
                measures the unintervened model while reporting an
                intervention, which is a false all-clear (axiom A7).
        """
        unknown = set(edits) - set(self._points)
        if unknown:
            raise KeyError(
                f"Unknown intervention point(s) {sorted(unknown)}. "
                f"Available: {sorted(self._points)}"
            )
        prev = self._edits
        self._edits = {**self._edits, **edits}
        self._fired -= set(edits)
        n_before = self.n_forwards
        try:
            yield self
        finally:
            self._edits = prev

        # Reached only on a clean exit, so an in-flight exception (which may
        # be why no forward ran) propagates untouched rather than being
        # replaced by a confusing report about a silent intervention.
        n_ran = self.n_forwards - n_before
        silent = sorted(set(edits) - self._fired)
        if n_ran and silent:
            raise InertIntervention(
                f"Intervention point(s) {silent} never fired during {n_ran} "
                f"forward pass(es), so the do-operation was a no-op and any "
                f"result measured inside the block reflects the unmodified "
                f"model. This usually means the module is invoked through a "
                f"method that bypasses __call__ (declare it in the adapter's "
                f"intervention_methods) or is not reached in the model's "
                f"current configuration (drop it from intervention_points)."
            )

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

    def batch_logits_with_trajectory(
        self, tokens: Sequence[Sequence[int]] | torch.Tensor
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Return ``(logits, [h_0, ..., h_L])`` for a batch.

        Requires ``Capabilities.has_hidden_states``. Each ``h_ℓ`` has shape
        ``(B, T, d)`` — the hidden state after layer ``ℓ``.
        """
        x = self._as_tokens(tokens)
        self.n_forwards += 1
        return self.adapter.forward_with_trajectory(self.model, x)

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
        """Remove hooks and method patches, and optionally restore the dtype."""
        for h in self._handles:
            h.remove()
        self._handles.clear()
        for module, method_name in self._patched:
            wrapper = module.__dict__.get(method_name)
            if wrapper is not None and hasattr(wrapper, "__scaf_wrapped__"):
                delattr(module, method_name)
        self._patched.clear()
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
