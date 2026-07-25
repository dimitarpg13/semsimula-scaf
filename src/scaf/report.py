"""``LeakScorecard`` — the audit record.

The scorecard's one opinionated rule: **a model is certified causal only if the
controls passed and every leak probe passed.** A probe that was skipped, or a
control that failed, yields ``INVALID`` rather than ``CLEAN``. There is no
outcome in which missing evidence reads as good news.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from .probes.base import ProbeResult

__all__ = ["LeakScorecard", "CausalLeakError"]


class CausalLeakError(AssertionError):
    """Raised by :meth:`LeakScorecard.assert_causal` when an audit fails."""


@dataclass
class LeakScorecard:
    """Aggregated result of an audit."""

    model: str = "unknown"
    adapter: str = "unknown"
    dtype: str = ""
    device: str = ""
    controls: list[ProbeResult] = field(default_factory=list)
    probes: list[ProbeResult] = field(default_factory=list)
    config: dict[str, Any] = field(default_factory=dict)
    notes: tuple[str, ...] = ()

    # ------------------------------------------------------------------
    @property
    def controls_ok(self) -> bool:
        return bool(self.controls) and all(c.passed is True for c in self.controls)

    @property
    def probes_ok(self) -> bool:
        return bool(self.probes) and all(p.passed is True for p in self.probes)

    @property
    def has_skips(self) -> bool:
        return any(r.skipped for r in self.controls + self.probes)

    @property
    def verdict(self) -> str:
        """``CLEAN`` / ``LEAK`` / ``INVALID``.

        ``INVALID`` outranks ``LEAK``: if the controls did not pass we cannot
        distinguish a clean model from a probe that never ran, so reporting
        either verdict would be a claim we have not earned.
        """
        if not self.controls_ok:
            return "INVALID"
        if any(p.passed is False for p in self.probes):
            return "LEAK"
        if self.has_skips or not self.probes:
            return "INVALID"
        return "CLEAN"

    @property
    def passed(self) -> bool:
        return self.verdict == "CLEAN"

    def get(self, name: str) -> ProbeResult | None:
        for r in self.controls + self.probes:
            if r.name == name:
                return r
        return None

    # ------------------------------------------------------------------
    def summary(self) -> str:
        head = (
            f"SCAF audit — {self.model}\n"
            f"  adapter={self.adapter} dtype={self.dtype} device={self.device}"
        )
        flags = {
            k: v for k, v in self.config.items()
            if k in ("prefix_causal_registers", "causal_force",
                     "reverse_channel", "n_registers", "fock_version")
        }
        if flags:
            head += "\n  flags: " + ", ".join(f"{k}={v}" for k, v in flags.items())

        lines = [head, "", "  Controls:"]
        lines += [f"    {c}" for c in self.controls]
        lines += ["", "  Probes:"]
        lines += [f"    {p}" for p in self.probes]

        tr = self.get("target_relocation")
        if tr and not tr.skipped:
            d = tr.detail
            mark = ">" if d.get("ppl_saturated") else ""
            lines += [
                "",
                f"    standard PPL {d['ppl_standard']:.4g}"
                f"  ->  honest PPL {mark}{d['ppl_honest']:.4g}"
                f"   ({tr.statistic:+.3f} nats,"
                f" {mark}{d['ppl_inflation']:.4g}x inflation)",
            ]
            if d.get("ppl_saturated"):
                lines.append(
                    "    (perplexities saturated; the nats gap is exact)"
                )

        for n in self.notes:
            lines.append(f"  note: {n}")

        lines += ["", f"  VERDICT: {self.verdict}"]
        if self.verdict == "INVALID":
            lines.append(
                "  (controls failed or probes were skipped — this audit "
                "cannot certify anything)"
            )
        return "\n".join(lines)

    def assert_causal(self) -> None:
        """Raise :class:`CausalLeakError` unless the verdict is ``CLEAN``."""
        if not self.passed:
            raise CausalLeakError(
                f"SCAF verdict {self.verdict}\n{self.summary()}"
            )

    # ------------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "adapter": self.adapter,
            "dtype": self.dtype,
            "device": self.device,
            "verdict": self.verdict,
            "config": self.config,
            "controls": [c.to_dict() for c in self.controls],
            "probes": [p.to_dict() for p in self.probes],
            "notes": list(self.notes),
        }

    def to_json(self) -> str:
        """Single-line JSON, ready to append to a training JSONL report."""
        return json.dumps(self.to_dict(), default=str)

    def __str__(self) -> str:
        return self.summary()
