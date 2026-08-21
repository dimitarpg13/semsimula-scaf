"""Basin-membership probe — Tier B geometric leak characterisation.

The test in one line: **change the future, and no past hidden state may
switch attractor wells.**

This probe escalates from Tier A's continuous cosine deviation to a
discrete, structurally meaningful quantity: whether the future perturbation
has moved a past hidden state from one attractor basin to another. A
basin-crossing leak is qualitatively more severe than a continuous
perturbation — it means the model is computing a *fundamentally different*
representation of the past, not just a slightly perturbed one.

The dominant well assignment is:

.. math::

    k^{\\ast}(h) = \\arg\\min_k \\bigl[
        d_{\\mathrm{Maha},k}^2(h) - 2\\log w_k
    \\bigr]

where :math:`d_{\\mathrm{Maha},k}^2 = (h - \\mu_k)^\\top \\Sigma_k^{-1} (h - \\mu_k)`
is the Mahalanobis distance to well *k*, evaluated via the diagonal +
low-rank decomposition
:math:`\\Sigma_k^{-1} = \\mathrm{diag}(a_k) + B_k B_k^\\top`
in :math:`O(d \\cdot r)` rather than :math:`O(d^2)`.

Requires ``Capabilities.has_vtheta_wells = True``. On adapters that do not
expose well parameters, the probe skips loudly.
"""

from __future__ import annotations

import torch

from .base import Probe, ProbeResult

__all__ = [
    "BasinMembershipProbe",
    "assign_dominant_wells",
    "measure_basin_crossings",
]


def assign_dominant_wells(
    h: torch.Tensor,
    mu: torch.Tensor,
    precision_diag: torch.Tensor,
    precision_lr: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """Assign each hidden state to its dominant Gaussian well.

    The dominant well is the one whose weighted Gaussian bump contributes
    most to :math:`V_\\theta(h)`, equivalently:

    .. math::

        k^{\\ast} = \\arg\\min_k \\bigl[
            (h - \\mu_k)^\\top \\Sigma_k^{-1} (h - \\mu_k) - 2\\log w_k
        \\bigr]

    Args:
        h: Hidden states, shape ``(B, T, d)`` or ``(B, d)``.
        mu: Well centres, shape ``(B, K, d)`` or ``(K, d)``.
        precision_diag: Diagonal precision ``a_k``, shape matching ``mu``.
        precision_lr: Low-rank factor ``B_k``, shape ``(..., K, d, r)``.
        weights: Mixture weights, shape ``(B, K)`` or ``(K,)``.

    Returns:
        Integer tensor of dominant well indices, shape ``(B, T)`` or ``(B,)``.
    """
    has_T = h.dim() == 3
    if has_T:
        # h: (B, T, d) → (B, T, 1, d)  for broadcasting against K wells
        h_e = h.unsqueeze(-2)
    else:
        # h: (B, d) → (B, 1, d)
        h_e = h.unsqueeze(-2)

    # Normalise well params to have a batch dim + a time dim so they
    # broadcast against h_e regardless of whether they arrived as
    # (K, d), (B, K, d), or already (B, 1, K, d).
    def _expand_kd(t: torch.Tensor) -> torch.Tensor:
        """Expand a (..., K, d) tensor to match h_e's leading dims."""
        if t.dim() == 2:
            # (K, d) → (1, K, d)
            t = t.unsqueeze(0)
        if has_T and t.dim() == 3:
            # (B, K, d) → (B, 1, K, d) — broadcast along T
            t = t.unsqueeze(1)
        return t

    def _expand_kdr(t: torch.Tensor) -> torch.Tensor:
        """Expand a (..., K, d, r) tensor to match h_e's leading dims."""
        if t.dim() == 3:
            t = t.unsqueeze(0)
        if has_T and t.dim() == 4:
            t = t.unsqueeze(1)
        return t

    def _expand_k(t: torch.Tensor) -> torch.Tensor:
        """Expand a (..., K) tensor to match the score leading dims."""
        if t.dim() == 1:
            t = t.unsqueeze(0)
        if has_T and t.dim() == 2:
            t = t.unsqueeze(1)
        return t

    mu = _expand_kd(mu)
    precision_diag = _expand_kd(precision_diag)
    precision_lr = _expand_kdr(precision_lr)

    diff = h_e - mu  # (..., K, d)

    # Diagonal part: sum_d a_k,d * (h_d - mu_k,d)^2
    diag_term = (precision_diag * diff * diff).sum(dim=-1)  # (..., K)

    # Low-rank part: ||B_k^T (h - mu_k)||^2
    Bt_diff = torch.einsum("...kd,...kdr->...kr", diff, precision_lr)
    lr_term = (Bt_diff * Bt_diff).sum(dim=-1)  # (..., K)

    mahalanobis_sq = diag_term + lr_term  # (..., K)

    # k* = argmin [d_Maha^2 - 2 log w_k]
    w = _expand_k(weights)
    log_w = torch.log(w.clamp(min=1e-30))
    score = mahalanobis_sq - 2.0 * log_w  # (..., K)
    return score.argmin(dim=-1)  # (...,)


#: Rank of each well-parameter tensor once a per-position time axis is
#: present: ``mu``/``precision_diag`` are ``(B, T, K, d)``, ``precision_lr``
#: adds the low-rank axis, ``weights`` drops ``d``.
_WELL_TIME_RANK = {
    "mu": 4, "precision_diag": 4, "precision_lr": 5, "weights": 3,
}


def _truncate_wells(wp: dict, t_end: int) -> dict:
    """Cut context-dependent wells down to the positions being scored.

    Wells derived from ``xi`` carry one landscape per position, so they must
    be sliced alongside the hidden states; context-free wells (the toys, and
    any model whose landscape is a bare parameter) have no time axis and are
    broadcast as-is.
    """
    return {
        k: (v[:, :t_end] if v.dim() == _WELL_TIME_RANK[k] else v)
        for k, v in wp.items()
    }


def measure_basin_crossings(
    im,
    x: torch.Tensor,
    pairs,
    micro_batch: int = 4,
) -> tuple[float, list[float], int, dict[str, float]]:
    """Measure basin-crossing rate between factual/counterfactual trajectories.

    Returns ``(crossing_rate, per_layer_rate, worst_layer, per_split_rate)``
    for a prepared perturbation pair set.

    ``crossing_rate`` is the fraction of (layer, position) pairs where the
    dominant well assignment changed — the Tier B analogue of AILE.
    """
    from .hidden_state import _chunked_trajectory

    _, base_traj = _chunked_trajectory(im, x, micro_batch)
    n_layers = len(base_traj)
    per_layer_crossings = [0] * n_layers
    per_layer_total = [0] * n_layers
    total_crossings = 0
    total_positions = 0
    per_split: dict[str, float] = {}

    for frac, t_p, x_cf in pairs:
        _, cf_traj = _chunked_trajectory(im, x_cf, micro_batch)

        for ell in range(n_layers):
            # Each arm is scored in its own landscape. The wells are a
            # function of xi, which is a function of the hidden state, so a
            # future perturbation moves the landscape as well as the point in
            # it. Holding the wells at their factual values would miss a leak
            # that travels through xi and would answer a different question
            # than "is this position in the same basin in both worlds".
            wp_f = im.adapter.well_parameters(im.model, ell, x, h=base_traj[ell])
            wp_c = im.adapter.well_parameters(
                im.model, ell, x_cf, h=cf_traj[ell]
            )
            if wp_f is None or wp_c is None:
                continue

            h_f = base_traj[ell][:, : t_p + 1]
            h_c = cf_traj[ell][:, : t_p + 1]
            wp_f = _truncate_wells(wp_f, t_p + 1)
            wp_c = _truncate_wells(wp_c, t_p + 1)

            wells_f = assign_dominant_wells(
                h_f, wp_f["mu"], wp_f["precision_diag"],
                wp_f["precision_lr"], wp_f["weights"],
            )
            wells_c = assign_dominant_wells(
                h_c, wp_c["mu"], wp_c["precision_diag"],
                wp_c["precision_lr"], wp_c["weights"],
            )

            crossed = (wells_f != wells_c).float()
            n_crossed = int(crossed.sum().item())
            n_total = crossed.numel()

            per_layer_crossings[ell] += n_crossed
            per_layer_total[ell] += n_total
            total_crossings += n_crossed
            total_positions += n_total

        key = f"crossing_rate_at_{frac:g}"
        if total_positions > 0:
            per_split[key] = total_crossings / total_positions

    crossing_rate = total_crossings / max(total_positions, 1)
    per_layer_rate = [
        c / max(t, 1) for c, t in zip(per_layer_crossings, per_layer_total)
    ]
    worst_layer = (
        int(max(range(n_layers), key=lambda i: per_layer_rate[i]))
        if n_layers > 0
        else 0
    )

    return crossing_rate, per_layer_rate, worst_layer, per_split


class BasinMembershipProbe(Probe):
    """Detect leaks via attractor-well reassignment in hidden states.

    This is the Tier B geometric probe. It reuses the same
    factual/counterfactual perturbation protocol as the Tier A probe but
    measures a discrete, structurally meaningful quantity: whether the
    dominant well assignment changes.

    Requires ``Capabilities.has_vtheta_wells = True``. On adapters that do
    not expose well parameters, the probe skips loudly.

    Args:
        splits: Fractions of the sequence at which to cut.
        n_seqs: Sequences per split point.
        n_pairs: Counterfactual futures per split.
        threshold: Maximum tolerated basin-crossing rate. ``0.0`` is correct
            for a structurally causal model: well assignments at ``t <= t_p``
            must not depend on tokens at ``> t_p``.
        micro_batch: Forward-pass chunk size; ``0`` disables chunking.
    """

    name = "basin_membership"

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
        if not im.caps.has_vtheta_wells:
            return self._skip(
                f"adapter {im.adapter.name!r} does not expose V_theta well "
                "parameters (has_vtheta_wells=False)"
            )

        from .future_perturbation import make_perturbation_pairs

        x, pairs = make_perturbation_pairs(
            corpus, self.n_seqs, self.splits, self.n_pairs, im.device
        )

        with im.deterministic():
            crossing_rate, per_layer, worst_layer, per_split = (
                measure_basin_crossings(im, x, pairs, self.micro_batch)
            )

        return ProbeResult(
            name=self.name,
            statistic=crossing_rate,
            unit="crossing_rate",
            threshold=self.threshold,
            passed=crossing_rate <= self.threshold,
            detail={
                "per_layer_crossing_rate": per_layer,
                "worst_layer": worst_layer,
                "n_layers": len(per_layer),
                "seq_len": int(x.shape[1]),
                "n_seqs": self.n_seqs,
                "n_pairs": self.n_pairs,
                **per_split,
            },
        )
