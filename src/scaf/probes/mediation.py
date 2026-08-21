"""Mediation probe — *where* is the leak, not *whether* there is one.

Once a leak is established, the operational question is which component
carries it, because that is what you have to change. The framework answers it
with the **controlled direct effect**: re-run the same future-perturbation
measurement with a candidate mediator clamped to a reference value.

.. math::

   \\mathrm{CDE}(M := m_0) = \\mathbb{E}\\big[ d\\big(
       L_t(\\mathrm{do\\ future},\\ \\mathrm{do}\\ M{=}m_0),\\
       L_t(\\mathrm{do}\\ M{=}m_0) \\big) \\big]

If the total effect is large while :math:`\\mathrm{CDE}(M{:=}0)` collapses to
zero, every path from the future to the past ran through :math:`M`, and
:math:`M` is the locus. The fraction removed is the attribution:

.. math::

   \\mathrm{attributed}(M) = 1 - \\frac{\\mathrm{CDE}(M := m_0)}{\\mathrm{Total}}

Two design points are load-bearing.

**Identical counterfactuals across arms.** Total and CDE are measured against
the very same perturbed futures, prepared once by
:func:`~scaf.probes.future_perturbation.make_perturbation_pairs`. Redrawing
between arms would let RNG variation masquerade as attribution.

**Knockout arithmetic is not free.** Clamping a parameter to zero is only a
knockout if zero is the *off* value. The Fock gate is
``scale = torch.tanh(reverse_channel_scale)``, so zero closes it — but a gate
behind a sigmoid would sit at half strength, and pushing an odd gate to a large
negative value merely flips its sign at full magnitude. SCAF verifies the
knockout empirically rather than assuming it: see ``knockout_verified`` below.

**Attribution is computed on the mean effect, not the max.** Detection uses
:math:`L_\\infty` because a causal model gives bit-exact zero there, needing no
tolerance. Attribution cannot: :math:`L_\\infty` is a max over positions, so
when two channels partially cancel at the single worst position, removing the
weaker one can *raise* it and yield a negative share. The CDE is defined as an
expectation, so the mean absolute effect (AILE) is the correct base — it
aggregates over all positions and behaves near-additively across channels.

A negative share survives that fix in one honest case: a component that was
**suppressing** the leak rather than carrying it. Those are reported as
suppressors instead of being clipped away.

This probe is a **diagnostic**. It does not decide whether a model is causal,
and it never contributes to the audit verdict — attribution quality must not be
able to make a leaky model look better.
"""

from __future__ import annotations

from ..core.intervenable import InertIntervention
from .base import Probe, ProbeResult
from .future_perturbation import (
    make_perturbation_pairs,
    measure_future_influence,
)

__all__ = ["MediationProbe"]


class MediationProbe(Probe):
    """Attribute a measured leak to the component that carries it.

    Args:
        mediators: Names to knock out. Defaults to the adapter's declared
            mediators, most-suspect first.
        splits: Sequence fractions at which to cut, as for the
            future-perturbation probe.
        n_seqs: Sequences to measure over.
        n_pairs: Counterfactual futures per split.
        micro_batch: Forward chunk size; ``0`` disables chunking.
        explain_threshold: Fraction of the total effect a single mediator must
            remove before it is called the locus.
        min_total: Smallest total effect worth attributing. Below this the
            probe skips: dividing a near-zero total into fractions produces
            impressive-looking percentages that are entirely noise.
    """

    name = "mediation"

    def __init__(
        self,
        mediators: tuple[str, ...] | None = None,
        splits: tuple[float, ...] = (0.25, 0.5, 0.75),
        n_seqs: int = 8,
        n_pairs: int = 2,
        micro_batch: int = 4,
        explain_threshold: float = 0.9,
        min_total: float = 1e-9,
    ) -> None:
        self.mediators = mediators
        self.splits = splits
        self.n_seqs = n_seqs
        self.n_pairs = n_pairs
        self.micro_batch = micro_batch
        self.explain_threshold = explain_threshold
        self.min_total = min_total

    # ------------------------------------------------------------------
    def run(self, im, corpus) -> ProbeResult:
        candidates = tuple(self.mediators or im.mediators)
        if not candidates:
            return self._skip(
                f"the {im.adapter.name} adapter declares no mediators, so "
                "there is nothing to knock out"
            )

        x, pairs = make_perturbation_pairs(
            corpus, self.n_seqs, self.splits, self.n_pairs, im.device
        )

        with im.deterministic():
            total_linf, total, _ = measure_future_influence(
                im, x, pairs, self.micro_batch
            )

            if total <= self.min_total:
                return self._skip(
                    f"no leak to attribute (mean effect {total:.3g} <= "
                    f"{self.min_total:g}); run mediation only on a model the "
                    "future-perturbation probe has already flagged"
                )

            per_mediator: dict[str, dict[str, float]] = {}
            unavailable: list[str] = []
            inert: list[str] = []
            for name in candidates:
                try:
                    with im.knockout(name):
                        cde_linf, cde, _ = measure_future_influence(
                            im, x, pairs, self.micro_batch
                        )
                except KeyError:
                    # Declared by capabilities but not actually intervenable.
                    unavailable.append(name)
                    continue
                except InertIntervention:
                    # The point exists and was hooked, but no forward pass
                    # reached it, so the "knockout" measured the untouched
                    # model. Its CDE would equal the total effect and the
                    # mediator would be scored at zero attribution —
                    # indistinguishable from a component that genuinely
                    # carries none of the leak. Record it as unmeasured.
                    inert.append(name)
                    continue
                per_mediator[name] = {
                    "cde": cde,
                    "cde_linf": cde_linf,
                    "attributed": 1.0 - cde / total,
                }

        if not per_mediator:
            return self._skip(
                f"none of the declared mediators {list(candidates)} could be "
                "knocked out on this model"
                + (f" ({inert} were hooked but never reached)" if inert else "")
            )

        ranked = sorted(
            per_mediator.items(), key=lambda kv: kv[1]["attributed"], reverse=True
        )
        top_name, top = ranked[0]

        # A component whose removal *increases* the leak was damping it, not
        # carrying it. Worth naming: it is a hint that the leak has more than
        # one channel and they partially cancel.
        suppressors = [n for n, m in ranked if m["attributed"] < -1e-9]

        detail: dict[str, object] = {
            "total": total,
            "total_linf": total_linf,
            "top_mediator": top_name,
            "ranking": [n for n, _ in ranked],
        }
        for name, m in per_mediator.items():
            detail[f"cde_{name}"] = m["cde"]
            detail[f"attributed_{name}"] = m["attributed"]
        if unavailable:
            detail["not_intervenable"] = unavailable
        if inert:
            # Not folded into not_intervenable: that list means "this model has
            # no such component", whereas these components exist and are
            # advertised as mediators while being unreachable through the
            # declared entry point. That is an adapter defect, and the
            # attribution below is over the remaining mediators only.
            detail["never_fired"] = inert
        if suppressors:
            detail["suppressors"] = suppressors
        # A knockout that fails to reduce the leak at all tells us the clamp
        # reference is wrong for this component, not that the component is
        # innocent. Surfacing it prevents a misread as exoneration.
        detail["knockout_verified"] = top["attributed"] > 0.0

        return ProbeResult(
            name=self.name,
            statistic=top["attributed"],
            unit="fraction",
            threshold=self.explain_threshold,
            passed=top["attributed"] >= self.explain_threshold,
            detail=detail,
        )
