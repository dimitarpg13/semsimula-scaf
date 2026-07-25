"""Controls that decide whether a probe's verdict can be believed.

A leak probe reports a number near zero. There are two reasons that can happen:
the model is causal, or **the probe never actually looked**. Nothing in the
number itself distinguishes them. The register leak survived an entire audit
cycle precisely because a clean result was read as the first when it was the
second.

So every audit runs three controls, and the scorecard refuses to certify a
model unless they pass:

* :class:`DeterminismControl` establishes the *noise floor*. If two identical
  forwards disagree, no leak threshold below that disagreement is meaningful.
* :class:`PlaceboControl` confirms the measurement pipeline reports zero when
  nothing was changed — catching bugs that would manufacture a false leak.
* :class:`PositiveControl` confirms the pipeline reports a large effect when
  something the model *is* allowed to see is changed. This is the one that
  would have caught the blind probe: it fails loudly when the intervention is
  not reaching the model.

Determinism and placebo guard against false alarms; the positive control guards
against false all-clears. Only the third protects against the failure mode that
actually cost this project a round of published numbers.
"""

from __future__ import annotations

import torch

from .probes.base import Probe, ProbeResult

__all__ = ["DeterminismControl", "PlaceboControl", "PositiveControl"]


class DeterminismControl(Probe):
    """Two identical forwards must produce identical logits.

    Args:
        n_seqs: Sequences to test.
        threshold: Tolerated disagreement. Zero by default — a stochastic
            forward makes every downstream comparison ambiguous, so we would
            rather fail here than silently widen tolerances elsewhere.
    """

    name = "control_determinism"

    def __init__(self, n_seqs: int = 4, threshold: float = 0.0) -> None:
        self.n_seqs = n_seqs
        self.threshold = threshold

    def run(self, im, corpus) -> ProbeResult:
        x = corpus.sample(self.n_seqs).to(im.device)
        with im.deterministic():
            a = im.batch_logits(x).float().cpu()
            b = im.batch_logits(x).float().cpu()
        floor = float((a - b).abs().max())
        return ProbeResult(
            name=self.name,
            statistic=floor,
            unit="logit",
            threshold=self.threshold,
            passed=floor <= self.threshold,
            detail={
                "note": (
                    "noise floor for all logit-space probes; leak thresholds "
                    "are raised to this value"
                ),
                "n_seqs": self.n_seqs,
            },
        )


class PlaceboControl(Probe):
    """A null intervention must produce exactly no change.

    The suffix is "replaced" with itself, so the probe's clone / chunk /
    compare machinery runs in full while the input is unchanged. A non-zero
    result here means the harness is fabricating differences, and any leak it
    reports is suspect.
    """

    name = "control_placebo"

    def __init__(self, n_seqs: int = 4, threshold: float = 0.0) -> None:
        self.n_seqs = n_seqs
        self.threshold = threshold

    def run(self, im, corpus) -> ProbeResult:
        x = corpus.sample(self.n_seqs).to(im.device)
        T = x.shape[1]
        t_p = T // 2
        x_null = x.clone()  # identity "perturbation"
        with im.deterministic():
            a = im.batch_logits(x).float().cpu()
            b = im.batch_logits(x_null).float().cpu()
        delta = float((a[:, : t_p + 1] - b[:, : t_p + 1]).abs().max())
        return ProbeResult(
            name=self.name,
            statistic=delta,
            unit="logit",
            threshold=self.threshold,
            passed=delta <= self.threshold,
            detail={"split": t_p, "n_seqs": self.n_seqs},
        )


class PositiveControl(Probe):
    """Perturbing a *visible* token must produce a large, detectable effect.

    This is the anti-blind-probe control. We change one token at position ``s``
    and check that logits at positions ``>= s`` — which are permitted, indeed
    required, to depend on it — actually move.

    A failure means the intervention is not reaching the model at all. In that
    state every other probe returns zero for a trivial reason, and a clean
    scorecard would be actively misleading. So a failure here invalidates the
    audit rather than merely annotating it.

    Args:
        min_effect: Smallest logit change accepted as evidence the probe is
            live. Deliberately loose — we are testing wiring, not effect size.
    """

    name = "control_positive"

    def __init__(
        self, n_seqs: int = 4, min_effect: float = 1e-4
    ) -> None:
        self.n_seqs = n_seqs
        self.min_effect = min_effect

    def run(self, im, corpus) -> ProbeResult:
        x = corpus.sample(self.n_seqs).to(im.device)
        T = x.shape[1]
        s = T // 2
        x_pert = corpus.perturb_position(x, s)
        if bool(torch.equal(x, x_pert)):
            return self._skip(
                "corpus could not produce a different token; vocabulary may "
                "be degenerate"
            )
        with im.deterministic():
            a = im.batch_logits(x).float().cpu()
            b = im.batch_logits(x_pert).float().cpu()
        # Positions >= s are the ones allowed to see the changed token.
        effect = float((a[:, s:] - b[:, s:]).abs().max())
        return ProbeResult(
            name=self.name,
            statistic=effect,
            unit="logit",
            threshold=self.min_effect,
            passed=effect >= self.min_effect,
            detail={
                "position": s,
                "n_seqs": self.n_seqs,
                "note": (
                    "must be LARGE; a small value means the probe is not "
                    "reaching the model and the audit is invalid"
                ),
            },
        )
