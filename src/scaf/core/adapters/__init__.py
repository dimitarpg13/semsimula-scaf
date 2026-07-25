"""Adapter registry and duck-typed auto-detection.

Resolution order:

1. An explicitly passed ``adapter=`` argument always wins.
2. A model-supplied ``__scaf_adapter__()`` hook, if present. This is the opt-in
   escape hatch for models that want to describe themselves; nothing is
   required to implement it.
3. Registered adapters whose ``detect()`` returns True, highest ``priority``
   first.

Step 3 never imports SemSimula code, so SCAF stays installable and testable
with no research repo present.
"""

from __future__ import annotations

from torch import nn

from .base import Capabilities, ModelAdapter, PositionalNode
from .fock import FockAdapter
from .generic import GenericAdapter

__all__ = [
    "Capabilities",
    "ModelAdapter",
    "PositionalNode",
    "FockAdapter",
    "GenericAdapter",
    "register_adapter",
    "resolve_adapter",
    "registered_adapters",
]

_REGISTRY: list[type[ModelAdapter]] = []


def register_adapter(cls: type[ModelAdapter]) -> type[ModelAdapter]:
    """Register an adapter class for auto-detection. Usable as a decorator."""
    if not issubclass(cls, ModelAdapter):
        raise TypeError(f"{cls!r} is not a ModelAdapter subclass")
    if cls not in _REGISTRY:
        _REGISTRY.append(cls)
    return cls


def registered_adapters() -> list[type[ModelAdapter]]:
    """Registered adapter classes, highest priority first."""
    return sorted(_REGISTRY, key=lambda c: c.priority, reverse=True)


def resolve_adapter(
    model: nn.Module, adapter: ModelAdapter | type[ModelAdapter] | None = None
) -> ModelAdapter:
    """Pick the adapter that should drive ``model``.

    Raises:
        TypeError: if ``model`` is not an ``nn.Module``.
        LookupError: if no adapter matches (only possible if the generic
            fallback has been deregistered).
    """
    if adapter is not None:
        if isinstance(adapter, ModelAdapter):
            return adapter
        if isinstance(adapter, type) and issubclass(adapter, ModelAdapter):
            return adapter()
        raise TypeError(f"adapter must be a ModelAdapter, got {adapter!r}")

    if not isinstance(model, nn.Module):
        raise TypeError(f"model must be a torch.nn.Module, got {type(model)!r}")

    hook = getattr(model, "__scaf_adapter__", None)
    if callable(hook):
        supplied = hook()
        if isinstance(supplied, ModelAdapter):
            return supplied
        if isinstance(supplied, type) and issubclass(supplied, ModelAdapter):
            return supplied()

    for cls in registered_adapters():
        try:
            if cls.detect(model):
                return cls()
        except Exception:  # noqa: BLE001 - a broken detector must not
            continue      # block the remaining adapters

    raise LookupError(
        f"No SCAF adapter matched {type(model).__name__}. Register one with "
        "scaf.register_adapter(), or pass adapter=GenericAdapter()."
    )


register_adapter(FockAdapter)
register_adapter(GenericAdapter)
