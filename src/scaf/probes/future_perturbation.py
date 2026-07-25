"""Future-perturbation probe — the Average Interventional Leak Effect (AILE).

The test in one line: **change the future, and nothing about the past may move.**

Formally, for a split point :math:`t_p` we compare a factual run against
:math:`\\mathrm{do}(x_{>t_p} := \\tilde x_{>t_p})` and measure how much the
logits at positions :math:`t \\le t_p` respond. In a correctly causal
autoregressive model position ``t`` is a function of ``x_0..x_t`` alone, so the
response is *exactly* zero — not small, zero. Any non-zero value is a directed
edge from the future into the past, which is the definition of a leak.

Two statistics are reported because they answer different questions:

* ``linf`` — the largest single logit deviation anywhere in the prefix. This is
  the *detector*: it is bit-exact zero for a causal model, so it answers "is
  there a leak at all?" with no tolerance tuning.
* ``aile`` — the mean absolute deviation. This is the *effect size*: it answers
  "how much does the leak actually move the model's beliefs?"

The reason both matter is that the original register leak had a *small* mean
effect under far-future perturbation while carrying an enormous in-window
advantage. Reporting only a mean is how a real leak gets dismissed as noise;
that is precisely what :class:`~scaf.probes.target_relocation.TargetRelocationProbe`
exists to cross-check.
"""

from __future__ import annotations

import torch

from .base import Probe, ProbeResult

__all__ = [
    "FuturePerturbationProbe",
    "PerturbationPairs",
    "make_perturbation_pairs",
    "measure_future_influence",
]


def _chunked_logits(im, x: torch.Tensor, micro_batch: int) -> torch.Tensor:
    """Run ``batch_logits`` in slices to bound peak memory.

    SemSimula forwards run with grad enabled (conservative forces need
    ``autograd.grad`` internally), so activation memory is retained during the
    pass even though no backward follows. On real checkpoints that is the
    difference between running and an OOM.

    Chunking must be numerically inert: it moves results to host memory but
    never changes their dtype. Downcasting here would silently destroy the
    bit-exact float64 baseline the architectural probe depends on.
    """
    if micro_batch <= 0 or x.shape[0] <= micro_batch:
        return im.batch_logits(x)
    outs = []
    for i in range(0, x.shape[0], micro_batch):
        outs.append(im.batch_logits(x[i: i + micro_batch]).cpu())
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return torch.cat(outs, dim=0)


#: ``(factual_tokens, [(split_fraction, t_p, counterfactual_tokens), ...])``.
PerturbationPairs = tuple[torch.Tensor, list[tuple[float, int, torch.Tensor]]]


def make_perturbation_pairs(
    corpus,
    n_seqs: int,
    splits: tuple[float, ...],
    n_pairs: int,
    device=None,
) -> PerturbationPairs:
    """Draw the factual batch and its counterfactual futures once.

    Materialising the interventions up front lets several measurements reuse
    *identical* counterfactuals. That is essential for mediation: a controlled
    direct effect is only comparable to the total effect if both were computed
    against the same perturbed futures. Redrawing between arms would fold RNG
    variation into the attribution and could manufacture, or hide, a mediator.
    """
    x = corpus.sample(n_seqs)
    if device is not None:
        x = x.to(device)
    T = x.shape[1]
    pairs: list[tuple[float, int, torch.Tensor]] = []
    for frac in splits:
        t_p = max(0, min(T - 2, int(round(frac * (T - 1)))))
        for _ in range(n_pairs):
            pairs.append((frac, t_p, corpus.perturb_suffix(x, t_p)))
    return x, pairs


def measure_future_influence(
    im, x: torch.Tensor, pairs, micro_batch: int = 4
) -> tuple[float, float, dict[str, float]]:
    """Return ``(linf, aile, per_split_linf)`` for a prepared pair set.

    Only positions ``<= t_p`` are compared: position ``t`` legitimately reads
    ``x_0..x_t``, so everything after the cut is free to move.
    """
    base = _chunked_logits(im, x, micro_batch)
    worst = 0.0
    total_abs, total_n = 0.0, 0
    per_split: dict[str, float] = {}

    for frac, t_p, x_cf in pairs:
        cf = _chunked_logits(im, x_cf, micro_batch)
        delta = (base[:, : t_p + 1] - cf[:, : t_p + 1]).abs()
        d_max = float(delta.max())
        key = f"linf_at_{frac:g}"
        per_split[key] = max(per_split.get(key, 0.0), d_max)
        worst = max(worst, d_max)
        total_abs += float(delta.sum())
        total_n += delta.numel()

    return worst, total_abs / max(total_n, 1), per_split


class FuturePerturbationProbe(Probe):
    """Measure whether future tokens influence past logits.

    Args:
        splits: Fractions of the sequence at which to cut. Several cuts are
            used because a leak can be range-limited — a register written at
            position ``s`` may only be readable a bounded distance back, so a
            single mid-sequence cut can miss it.
        n_seqs: Sequences per split point.
        n_pairs: Independent counterfactual futures drawn per split. More
            draws lower the chance that a leak is missed because one random
            future happened to be uninformative.
        threshold: Maximum tolerated ``linf``. Defaults to ``0.0``: for a
            structurally causal model the prefix computation is untouched by
            suffix values, so bit-exact equality is the correct expectation.
            :func:`scaf.audit` raises this to the measured determinism floor
            when the platform is not bit-reproducible.
        micro_batch: Forward-pass chunk size; ``0`` disables chunking.
    """

    name = "future_perturbation"

    def __init__(
        self,
        splits: tuple[float, ...] = (0.25, 0.5, 0.75),
        n_seqs: int = 8,
        n_pairs: int = 2,
        threshold: float = 0.0,
        micro_batch: int = 4,
    ) -> None:
        self.splits = splits
        self.n_seqs = n_seqs
        self.n_pairs = n_pairs
        self.threshold = threshold
        self.micro_batch = micro_batch

    def run(self, im, corpus) -> ProbeResult:
        x, pairs = make_perturbation_pairs(
            corpus, self.n_seqs, self.splits, self.n_pairs, im.device
        )
        with im.deterministic():
            linf, aile, per_split = measure_future_influence(
                im, x, pairs, self.micro_batch
            )

        return ProbeResult(
            name=self.name,
            statistic=linf,
            unit="logit",
            threshold=self.threshold,
            passed=linf <= self.threshold,
            detail={
                "aile": aile,
                "seq_len": int(x.shape[1]),
                "n_seqs": self.n_seqs,
                "n_pairs": self.n_pairs,
                **per_split,
            },
        )
