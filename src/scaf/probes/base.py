"""``ProbeResult`` — the single record type every probe and control returns.

The design point worth stating: ``passed`` is a *tri-state*. ``True`` and
``False`` are ordinary, but ``None`` means "this probe did not run", and it is
never silently coerced to a pass. A probe that cannot run — because the model
lacks the feature, or because a prerequisite control failed — must be visibly
inconclusive in the scorecard.

That distinction is the whole lesson of the register-leak episode: the original
audit reported a clean bill of health from a probe that was not, in fact,
exercising the leak pathway. An audit framework that cannot say "I don't know"
will eventually say "fine" when it means "I didn't look".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

__all__ = ["ProbeResult", "Probe"]


@dataclass
class ProbeResult:
    """Outcome of a single probe or control.

    Attributes:
        name: Probe identifier, used as the scorecard row label.
        statistic: The measured quantity. Meaning depends on ``unit``.
        unit: What ``statistic`` is measured in — ``"logit"`` for a max
            absolute logit deviation, ``"nats"`` for an NLL gap, ``"bool"`` for
            a pass/fail control.
        threshold: Tolerance the statistic was compared against. ``None`` when
            the probe is purely descriptive.
        passed: ``True`` / ``False`` / ``None`` for did-not-run.
        skipped_reason: Why the probe did not run. Required when ``passed`` is
            ``None``, so the scorecard can always explain a gap.
        detail: Probe-specific extras, surfaced in ``to_dict`` for JSONL logs.
    """

    name: str
    statistic: float = 0.0
    unit: str = "logit"
    threshold: float | None = None
    passed: bool | None = None
    skipped_reason: str | None = None
    detail: dict[str, Any] = field(default_factory=dict)

    @property
    def skipped(self) -> bool:
        return self.passed is None

    @property
    def status(self) -> str:
        if self.passed is None:
            return "SKIP"
        return "PASS" if self.passed else "FAIL"

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "statistic": self.statistic,
            "unit": self.unit,
            "threshold": self.threshold,
            "skipped_reason": self.skipped_reason,
            **{f"detail_{k}": v for k, v in self.detail.items()},
        }

    def __str__(self) -> str:
        if self.passed is None:
            return f"[SKIP] {self.name}: {self.skipped_reason}"
        cmp = "" if self.threshold is None else f" (tol {self.threshold:g})"
        return (
            f"[{self.status}] {self.name}: "
            f"{self.statistic:.6g} {self.unit}{cmp}"
        )


class Probe:
    """Base class for probes. Subclasses implement :meth:`run`."""

    name: str = "probe"

    def run(self, im, corpus) -> ProbeResult:  # pragma: no cover - interface
        raise NotImplementedError

    def _skip(self, reason: str) -> ProbeResult:
        return ProbeResult(name=self.name, skipped_reason=reason)
