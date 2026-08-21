"""Hidden-state cosine-deviation probe — Tier A geometric leak detection.

The test in one line: **change the future, and no hidden state in the past
may rotate.**

This is the hidden-state analogue of
:class:`~scaf.probes.future_perturbation.FuturePerturbationProbe`. Where that
probe measures whether *logits* respond to a future perturbation, this one
measures whether *per-layer hidden states* respond. The metric is cosine
deviation:

.. math::

    \\Delta_{\\cos}^{(\\ell)}(t)
      = 1 - \\cos\\bigl(h_\\ell^{(t)}[\\text{factual}],\\,
                         h_\\ell^{(t)}[\\text{counterfactual}]\\bigr)

Why cosine, not L2: the SemSimula hidden-state space carries a Jacobi metric
that is conformally flat — :math:`\\tilde g = \\Omega^2 g`. Under any conformal
rescaling the :math:`\\Omega` factors cancel in the cosine ratio, so cosine
deviation is the same quantity whether measured in flat coordinates or in the
model's curved Riemannian geometry. L2 distance would conflate leak magnitude
with the local potential energy.

Why it matters beyond logit L∞: a hidden-state leak that hasn't yet
propagated to logits — e.g. register states carrying future information that a
subsequent LayerNorm washes out of the magnitude, or an output projection that
nulls the leaked component — shows :math:`\\Delta_{\\cos} > 0` even when logit
L∞ is exactly zero. This is a **latent leak**: the model's internal
representation of the past is corrupted, but the output happens to mask it.
That is a fragile invariant, not a guarantee.

The per-layer profile :math:`\\Delta_{\\cos}^{(\\ell)}` is a diagnostic
fingerprint. A spike at the layer where the reverse channel injects, decaying
through subsequent layers, identifies the Fock reverse-channel mechanism. A
spike at the embedding layer growing through later layers identifies a masking
or wiring bug.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .base import Probe, ProbeResult

__all__ = [
    "HiddenStateLeakProbe",
    "measure_hidden_state_influence",
]


def _chunked_trajectory(im, x: torch.Tensor, micro_batch: int):
    """Run ``batch_logits_with_trajectory`` in slices to bound peak memory.

    Returns ``(logits, trajectories)`` where each element of ``trajectories``
    is a ``(B, T, d)`` tensor on CPU.
    """
    if micro_batch <= 0 or x.shape[0] <= micro_batch:
        logits, traj = im.batch_logits_with_trajectory(x)
        return logits, [h.cpu() for h in traj]

    all_logits = []
    all_trajs: list[list[torch.Tensor]] = []
    for i in range(0, x.shape[0], micro_batch):
        logits, traj = im.batch_logits_with_trajectory(x[i: i + micro_batch])
        all_logits.append(logits.cpu())
        if not all_trajs:
            all_trajs = [[] for _ in traj]
        for layer_idx, h in enumerate(traj):
            all_trajs[layer_idx].append(h.cpu())
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return (
        torch.cat(all_logits, dim=0),
        [torch.cat(chunks, dim=0) for chunks in all_trajs],
    )


def _cosine_deviation(h_a: torch.Tensor, h_b: torch.Tensor) -> torch.Tensor:
    """Compute ``1 - cos(h_a, h_b)`` along the last dimension.

    Returns a tensor of the same leading shape as ``h_a[:, :, ...]``.

    Bit-identical inputs are forced to exactly zero. A cosine is a ratio of
    two separately-accumulated reductions, so ``cos(h, h)`` comes back as
    ``1 ± eps`` rather than exactly 1 — around ``2e-7`` in float32 for a
    few hundred dimensions. The probe's threshold is an exact zero, because
    a causal model must produce bit-identical past states, so that residue
    alone would fail every model ever tested, clean ones included. Masking it
    keeps the exact-zero threshold meaningful: a position whose two hidden
    states agree bit-for-bit has, by definition, not deviated.

    Near-identical states are clamped at zero for the same reason: the same
    rounding puts ``cos`` marginally above 1, and a *negative* deviation is
    not a physical quantity — it is noise with a sign, which a downstream
    CATE regression would happily fit.
    """
    dev = 1.0 - F.cosine_similarity(h_a, h_b, dim=-1)
    dev = dev.masked_fill((h_a == h_b).all(dim=-1), 0.0)
    return dev.clamp_min(0.0)


def measure_hidden_state_influence(
    im,
    x: torch.Tensor,
    pairs,
    micro_batch: int = 4,
) -> tuple[float, list[float], dict[str, float]]:
    """Measure cosine deviation between factual/counterfactual hidden states.

    Returns ``(max_delta_cos, per_layer_worst, per_split_max)`` for a
    prepared perturbation pair set (from
    :func:`~scaf.probes.future_perturbation.make_perturbation_pairs`).

    ``max_delta_cos`` is the worst-case cosine deviation across all layers,
    positions, and splits — the hidden-state analogue of ``linf``.

    ``per_layer_worst`` is a list of per-layer worst-case deviations — the
    diagnostic profile that fingerprints the leak mechanism.
    """
    _, base_traj = _chunked_trajectory(im, x, micro_batch)
    n_layers = len(base_traj)
    per_layer_worst = [0.0] * n_layers
    overall_worst = 0.0
    per_split: dict[str, float] = {}

    for frac, t_p, x_cf in pairs:
        _, cf_traj = _chunked_trajectory(im, x_cf, micro_batch)
        for ell in range(n_layers):
            dev = _cosine_deviation(
                base_traj[ell][:, : t_p + 1],
                cf_traj[ell][:, : t_p + 1],
            )
            d_max = float(dev.max())
            per_layer_worst[ell] = max(per_layer_worst[ell], d_max)
            overall_worst = max(overall_worst, d_max)

        key = f"max_cos_dev_at_{frac:g}"
        per_split[key] = max(per_split.get(key, 0.0), overall_worst)

    return overall_worst, per_layer_worst, per_split


class HiddenStateLeakProbe(Probe):
    """Detect leaks via cosine deviation in per-layer hidden states.

    This is the Tier A geometric probe. It reuses the same
    factual/counterfactual perturbation protocol as
    :class:`~scaf.probes.future_perturbation.FuturePerturbationProbe`
    but measures cosine deviation in hidden-state space rather than absolute
    deviation in logit space.

    Requires ``Capabilities.has_hidden_states = True``. On adapters that do
    not support trajectory extraction, the probe skips loudly.

    Args:
        splits: Fractions of the sequence at which to cut.
        n_seqs: Sequences per split point.
        n_pairs: Counterfactual futures per split.
        threshold: Maximum tolerated ``max_delta_cos``. ``0.0`` is correct
            for a structurally causal model: hidden states at ``t <= t_p``
            must not depend on tokens at ``> t_p``.
        micro_batch: Forward-pass chunk size; ``0`` disables chunking.
    """

    name = "hidden_state_leak"

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
        if not im.caps.has_hidden_states:
            return self._skip(
                f"adapter {im.adapter.name!r} does not expose hidden states "
                "(has_hidden_states=False)"
            )

        from .future_perturbation import make_perturbation_pairs

        x, pairs = make_perturbation_pairs(
            corpus, self.n_seqs, self.splits, self.n_pairs, im.device
        )

        with im.deterministic():
            # Also run the logit-level comparison to detect latent leaks.
            from .future_perturbation import measure_future_influence

            linf, _aile, _per_split = measure_future_influence(
                im, x, pairs, self.micro_batch
            )

            max_cos, per_layer, per_split = measure_hidden_state_influence(
                im, x, pairs, self.micro_batch
            )

        latent_leak = max_cos > self.threshold and linf <= self.threshold
        peak_layer = (
            int(max(range(len(per_layer)), key=lambda i: per_layer[i]))
            if per_layer
            else 0
        )

        return ProbeResult(
            name=self.name,
            statistic=max_cos,
            unit="cosine_dev",
            threshold=self.threshold,
            passed=max_cos <= self.threshold,
            detail={
                "per_layer_delta_cos": per_layer,
                "peak_layer": peak_layer,
                "n_layers": len(per_layer),
                "latent_leak": latent_leak,
                "logit_linf": linf,
                "seq_len": int(x.shape[1]),
                "n_seqs": self.n_seqs,
                "n_pairs": self.n_pairs,
                **per_split,
            },
        )
