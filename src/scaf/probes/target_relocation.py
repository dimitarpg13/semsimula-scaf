"""Target-relocation probe — honest perplexity and the leak tax in nats.

Where :class:`~scaf.probes.future_perturbation.FuturePerturbationProbe` asks
*"can the future reach the past?"*, this probe asks the question that actually
decides whether a reported number is publishable: **how much of the model's
perplexity is paid for by information it should not have had?**

The method is to score the same target token twice:

* **standard** — feed the whole sequence, read the logit at position ``t``.
  This is how perplexity is conventionally computed, and it is what every
  reported PPL in the project used.
* **honest** — feed only ``x_0..x_t``, read the logit at position ``t``. The
  future physically cannot be consulted because it was never supplied.

For a causal model the two are the same computation and agree to within kernel
noise. The gap

.. math::  \\tau_{\\text{leak}} = \\mathrm{NLL}_{\\text{honest}}
                                - \\mathrm{NLL}_{\\text{standard}}

is the leak tax in nats. On the pre-fix d=384 checkpoint this probe is what
turned a celebrated 7.69 PPL into an honest 258.07 — a gap of +3.51 nats that
the future-perturbation test alone had understated, because far-future topic
swaps move the logits far less than the immediate next-token context the shared
register state was smuggling in.

**On tolerances.** Unlike the future-perturbation probe, bit-exact zero is *not*
the right expectation here. Truncating the input changes the sequence-length
dimension, which changes matmul tiling and reduction order, so even a perfectly
causal model shows float-level disagreement. The default threshold is therefore
a small positive epsilon, chosen orders of magnitude below any leak worth
caring about.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .base import Probe, ProbeResult

__all__ = ["TargetRelocationProbe"]

#: NLL above which exponentiating to a perplexity stops being informative.
#: The gap in nats is always the exact, primary statistic; the perplexities are
#: a convenience rendering of it, and saturating them keeps a pathological
#: model from producing a meaningless 10^26 in the report.
_MAX_NLL_FOR_PPL = 40.0


def _ppl(nats: float) -> float:
    return math.exp(min(nats, _MAX_NLL_FOR_PPL))


class TargetRelocationProbe(Probe):
    """Compare standard perplexity against leak-free ("honest") perplexity.

    Args:
        n_seqs: Sequences to average over.
        n_targets: Target positions per sequence. Each costs one extra forward
            pass, so this is the probe's main cost knob.
        first_frac: Earliest target position as a fraction of the sequence.
            Very early targets have little context and inflate both NLLs
            without telling us anything about the gap.
        threshold: Maximum tolerated ``tau_leak`` in nats.
        micro_batch: Forward-pass chunk size; ``0`` disables chunking.
    """

    name = "target_relocation"

    def __init__(
        self,
        n_seqs: int = 8,
        n_targets: int = 32,
        first_frac: float = 0.25,
        threshold: float = 1e-3,
        micro_batch: int = 4,
        min_targets: int = 4,
    ) -> None:
        self.n_seqs = n_seqs
        self.n_targets = n_targets
        self.first_frac = first_frac
        self.threshold = threshold
        self.micro_batch = micro_batch
        self.min_targets = min_targets

    # ------------------------------------------------------------------
    def _nll_at(self, logits_t: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Per-row NLL of ``targets`` under ``logits_t`` of shape ``(B, V)``.

        Half precisions are promoted to float32 for numerical stability, but
        float64 is preserved — the whole point of a float64 audit is that the
        reported gap is not an artefact of rounding.
        """
        dt = torch.promote_types(logits_t.dtype, torch.float32)
        return F.cross_entropy(logits_t.to(dt), targets, reduction="none")

    def _forward_chunked(self, im, x: torch.Tensor) -> torch.Tensor:
        """Chunk for memory only; dtype is preserved exactly."""
        mb = self.micro_batch
        if mb <= 0 or x.shape[0] <= mb:
            return im.batch_logits(x)
        outs = []
        for i in range(0, x.shape[0], mb):
            outs.append(im.batch_logits(x[i: i + mb]).cpu())
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return torch.cat(outs, dim=0)

    # ------------------------------------------------------------------
    def run(self, im, corpus) -> ProbeResult:
        x = corpus.sample(self.n_seqs).to(im.device)
        T = x.shape[1]

        lo = max(1, int(round(self.first_frac * (T - 1))))
        hi = T - 2
        available = hi - lo + 1
        if available < self.min_targets:
            # Averaging a leak tax over one or two positions is not a
            # measurement. Report the gap in evidence instead of a number
            # nobody should act on.
            return self._skip(
                f"sequence too short: {max(available, 0)} scoreable positions "
                f"at T={T}, need {self.min_targets}"
            )
        n = min(self.n_targets, available)
        positions = torch.linspace(lo, hi, steps=n).round().long().tolist()
        positions = sorted(set(positions))

        with im.deterministic():
            full = self._forward_chunked(im, x)

            std_nll, honest_nll = [], []
            for t in positions:
                targets = x[:, t + 1].to(full.device)
                std_nll.append(self._nll_at(full[:, t], targets))

                # Truncation is the intervention: the future is not masked,
                # it is absent, so no implementation detail can leak it.
                trunc = self._forward_chunked(im, x[:, : t + 1])
                honest_nll.append(self._nll_at(trunc[:, t], targets))

        std = torch.cat(std_nll).mean().item()
        honest = torch.cat(honest_nll).mean().item()
        tau = honest - std

        return ProbeResult(
            name=self.name,
            statistic=tau,
            unit="nats",
            threshold=self.threshold,
            passed=abs(tau) <= self.threshold,
            detail={
                "nll_standard": std,
                "nll_honest": honest,
                "ppl_standard": _ppl(std),
                "ppl_honest": _ppl(honest),
                "ppl_inflation": _ppl(tau),
                "ppl_saturated": max(std, honest, tau) > _MAX_NLL_FOR_PPL,
                "n_targets": len(positions),
                "n_seqs": self.n_seqs,
                "seq_len": T,
            },
        )
