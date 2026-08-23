"""Stiffness probe — Tier C dynamical-stability audit.

The test in one line: **at the hidden states this checkpoint's own forward
pass actually visits, is the local curvature small enough for its own step
size to remain numerically stable?**

Velocity-Verlet (and the explicit-Euler family more broadly) is stable for a
harmonic oscillator of angular frequency :math:`\\omega` and step size
:math:`\\Delta t` if and only if :math:`\\omega \\Delta t \\lt 2`. A
conservative-force SemSimula layer step is, at any frozen hidden state
:math:`h`, locally a harmonic oscillator with stiffness :math:`K(h) =
-\\partial^2 V_\\theta / \\partial h^2` and mass :math:`\\mathfrak{m}(h)`, so
:math:`\\omega(h) = \\sqrt{K(h) / \\mathfrak{m}(h)}` and the same bound
applies pointwise. See the companion notes on symplectic integration and the
PyTorch CfC/BAOAB implementation deep dive for the derivation.

This probe samples that quantity across a validation corpus, at the *native*
hidden-state trajectory the checkpoint's own configured integrator actually
produces — :meth:`~scaf.core.intervenable.InterventableModel.batch_logits_with_trajectory`
never overrides ``cfg.integrator``, unlike the notebook-only predecessor of
this probe, which had to force ``cfg.integrator = 'baoab_cfc'`` for the
duration of the probe because ``harmonic_terms()`` used to only ever be
called from the CfC layer step. Trajectory fidelity here is a byproduct of
reusing the same adapter primitive every other trajectory-based probe
(:class:`~scaf.probes.hidden_state.HiddenStateLeakProbe`,
:class:`~scaf.probes.basin_membership.BasinMembershipProbe`) already uses,
not a special case that needs its own caveat.

Two curvature estimates are reported, gated on independent capability flags:

* ``k_diag`` (requires ``Capabilities.has_harmonic_terms``): the model's own
  closed-form harmonic linearisation, exact for some ``V_theta`` families
  (structured quadratic wells) and a diagonal proxy for others (Gaussian
  mixtures, where the true precision matrix has off-diagonal structure
  ``k_diag`` cannot see).
* A Weyl-inequality upper bound on the true top eigenvalue (requires
  ``Capabilities.has_vtheta_wells``, so only available for Gaussian-mixture
  families): :math:`\\lambda_{\\max}(P_k) \\le \\max_i a_k[i] +
  \\sigma_{\\max}(B_k)^2`, aggregated the same way ``k_diag`` aggregates
  across wells. This is a **conservative, any-direction** certificate — it
  can only overstate the true worst-case curvature, never understate it —
  computed purely from :meth:`~scaf.core.adapters.base.ModelAdapter.well_parameters`'s
  existing output, with no new adapter surface needed.
"""

from __future__ import annotations

import numpy as np
import torch

from .base import Probe, ProbeResult
from .hidden_state import _chunked_trajectory

__all__ = ["StiffnessProbe"]

#: Floor on k/m before the sqrt, matching cfc_baoab.py's _OMEGA_SQ_FLOOR.
#: torch.sqrt has an infinite derivative at exactly 0, and float32 underflow
#: of the Gaussian bump weights can drive k_diag to exact 0.0 in practice —
#: this probe is read-only (no backward pass), but the floor keeps its output
#: consistent with the training-time fix and avoids a spurious inf at 0/0.
_OMEGA_SQ_FLOOR = 1e-12


def _omega_dt(curvature: torch.Tensor, mass: torch.Tensor | None, dt: float) -> torch.Tensor:
    """Turn a curvature sample into the dimensionless quantity omega*dt.

    ``mass`` may carry one more trailing singleton dimension than
    ``curvature`` (per-position mass broadcasting against a per-dimension
    curvature) or be a 0-d scalar (global mass mode); both are squeezed to
    broadcast correctly.
    """
    if mass is None:
        m = torch.ones((), dtype=curvature.dtype, device=curvature.device)
    else:
        m = mass.to(dtype=curvature.dtype, device=curvature.device)
        while m.dim() > curvature.dim():
            m = m.squeeze(-1)
    omega = (curvature.clamp(min=0.0) / m.clamp(min=_OMEGA_SQ_FLOOR)).sqrt()
    return omega * dt


def _quantiles(x: torch.Tensor) -> dict[str, float]:
    """Median / p90 / p99 / p99.9 / max of a flat tensor."""
    if x.numel() == 0:
        return {"median": 0.0, "p90": 0.0, "p99": 0.0, "p999": 0.0, "max": 0.0}
    xf = x.detach().to(dtype=torch.float64).cpu()
    try:
        q = torch.quantile(xf, torch.tensor([0.5, 0.9, 0.99, 0.999], dtype=torch.float64))
        median, p90, p99, p999 = (float(v) for v in q)
    except RuntimeError:
        # torch.quantile refuses tensors above its internal sort-based size
        # limit; numpy's percentile has no such ceiling.
        median, p90, p99, p999 = (
            float(v) for v in np.percentile(xf.numpy(), [50, 90, 99, 99.9])
        )
    return {"median": median, "p90": p90, "p99": p99, "p999": p999, "max": float(xf.max())}


def _frac(x: torch.Tensor, threshold: float) -> float:
    if x.numel() == 0:
        return 0.0
    return float((x > threshold).to(dtype=torch.float64).mean())


def _bootstrap_frac_ci(
    blocks: list[torch.Tensor],
    threshold: float,
    n_boot: int = 500,
    seed: int = 0,
) -> tuple[float, float] | None:
    """95% block-bootstrap CI on frac(x > threshold).

    Resamples whole blocks — each an independently-drawn corpus batch — with
    replacement, rather than individual scalars. Scalars within one batch
    share a hidden-state trajectory and are therefore correlated; a
    per-scalar bootstrap would understate the true sampling uncertainty.

    Returns ``None`` when fewer than two non-empty blocks are available, so
    a single-batch probe run degrades to a point estimate with no CI rather
    than a spuriously narrow one.
    """
    blocks = [b for b in blocks if b.numel() > 0]
    if len(blocks) < 2:
        return None
    counts = np.array([b.numel() for b in blocks], dtype=np.float64)
    exceed = np.array(
        [(b > threshold).sum().item() for b in blocks], dtype=np.float64
    )
    rng = np.random.default_rng(seed)
    n = len(blocks)
    boot = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot[i] = exceed[idx].sum() / counts[idx].sum()
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return float(lo), float(hi)


def _expand_kd(t: torch.Tensor, has_T: bool) -> torch.Tensor:
    if t.dim() == 2:
        t = t.unsqueeze(0)
    if has_T and t.dim() == 3:
        t = t.unsqueeze(1)
    return t


def _expand_kdr(t: torch.Tensor, has_T: bool) -> torch.Tensor:
    if t.dim() == 3:
        t = t.unsqueeze(0)
    if has_T and t.dim() == 4:
        t = t.unsqueeze(1)
    return t


def _expand_k(t: torch.Tensor, has_T: bool) -> torch.Tensor:
    if t.dim() == 1:
        t = t.unsqueeze(0)
    if has_T and t.dim() == 2:
        t = t.unsqueeze(1)
    return t


def weyl_upper_bound(h: torch.Tensor, well_params: dict[str, torch.Tensor]) -> torch.Tensor:
    """Aggregate Weyl-inequality upper bound K_Weyl(h) on the true top
    eigenvalue of the effective precision matrix sum_k g_k P_k.

    Recomputes the same Gaussian-bump weights ``g_k`` every
    ``harmonic_terms()`` implementation for this family already computes
    internally, from the public ``well_parameters()`` output alone — no new
    adapter method is needed. ``P_k = diag(a_k) + B_k B_k^T``, so by Weyl's
    inequality ``lambda_max(P_k) <= max_i a_k[i] + sigma_max(B_k)^2``, and
    applying it a second time across wells gives the aggregate returned
    here. See ``docs/Example_Stiffness_Audit_OWT_g0.1_Anisotropic_Gaussian.md``
    (semsimula-scaf) section 5 for the full derivation.

    Args:
        h: Hidden states, shape ``(B, T, d)`` or ``(B, d)``.
        well_params: The dict returned by
            :meth:`~scaf.core.adapters.base.ModelAdapter.well_parameters`.

    Returns:
        ``K_Weyl(h)``, shape matching ``h`` minus its last dimension.
    """
    mu, a, B, w = (
        well_params["mu"], well_params["precision_diag"],
        well_params["precision_lr"], well_params["weights"],
    )
    has_T = h.dim() == 3
    h_e = h.unsqueeze(-2)

    mu = _expand_kd(mu, has_T)
    a = _expand_kd(a, has_T)
    B = _expand_kdr(B, has_T)
    w = _expand_k(w, has_T)

    diff = h_e - mu
    diag_term = (a * diff * diff).sum(dim=-1)
    Bt_diff = torch.einsum("...kd,...kdr->...kr", diff, B)
    lr_term = (Bt_diff * Bt_diff).sum(dim=-1)
    g = w * torch.exp(-0.5 * (diag_term + lr_term))  # (..., K)

    rank = B.shape[-1]
    if rank == 0:
        sigma_max_sq = torch.zeros_like(a[..., 0])
    else:
        gram = torch.einsum("...kdr,...kds->...krs", B, B)  # (..., K, r, r)
        sigma_max_sq = torch.linalg.eigvalsh(gram)[..., -1]  # (..., K)

    per_well = a.max(dim=-1).values + sigma_max_sq  # (..., K)
    return (g * per_well).sum(dim=-1)  # (...,)


class StiffnessProbe(Probe):
    """Verlet/CfC stability audit via the model's own harmonic linearisation.

    Requires ``Capabilities.has_harmonic_terms = True``. On adapters that do
    not expose a closed-form curvature (e.g. the plain MLP V_theta family),
    the probe skips loudly rather than silently reporting a clean bill of
    health it never actually measured.

    Args:
        stability_bound: omega*dt above this is unstable for an explicit
            symplectic integrator (2.0, from the harmonic-oscillator bound).
        marginal_bound: omega*dt above this is an early-warning zone,
            reported as ``frac_marginal`` but not counted as unstable.
        frac_unstable_threshold: tolerated fraction of unstable samples.
            ``0.0`` is the correct value for a run that never crosses the
            bound anywhere it was sampled.
        n_batches: number of independently-drawn corpus batches. Each is one
            resampling unit for the block-bootstrap CI — more batches gives
            a tighter CI, not just more samples per batch.
        seqs_per_batch: sequences per batch.
        micro_batch: forward-pass chunk size within a batch, to bound peak
            memory; ``0`` disables chunking.
        n_boot: bootstrap resamples for the 95% CI.
        bootstrap_seed: seed for the bootstrap RNG, independent of the
            corpus's own seed, so re-running the probe with a different
            ``n_boot`` does not perturb which batches were sampled.
    """

    name = "stiffness"

    def __init__(
        self,
        stability_bound: float = 2.0,
        marginal_bound: float = 1.0,
        frac_unstable_threshold: float = 0.0,
        n_batches: int = 4,
        seqs_per_batch: int = 2,
        micro_batch: int = 4,
        n_boot: int = 500,
        bootstrap_seed: int = 0,
    ) -> None:
        self.stability_bound = stability_bound
        self.marginal_bound = marginal_bound
        self.frac_unstable_threshold = frac_unstable_threshold
        self.n_batches = n_batches
        self.seqs_per_batch = seqs_per_batch
        self.micro_batch = micro_batch
        self.n_boot = n_boot
        self.bootstrap_seed = bootstrap_seed

    def run(self, im, corpus) -> ProbeResult:
        if not im.caps.has_harmonic_terms:
            return self._skip(
                f"adapter {im.adapter.name!r} does not expose harmonic "
                "curvature (has_harmonic_terms=False)"
            )

        dt = im.config().get("dt")
        if not dt:
            return self._skip(
                f"adapter {im.adapter.name!r} did not report a step size "
                "('dt' missing from config()); omega*dt cannot be formed"
            )
        dt = float(dt)

        has_weyl = im.caps.has_vtheta_wells
        omega_dt_blocks: list[torch.Tensor] = []
        eig_omega_dt_blocks: list[torch.Tensor] = []
        layers_seen: set[int] = set()

        with im.deterministic():
            for _ in range(self.n_batches):
                x = corpus.sample(self.seqs_per_batch).to(im.device)
                _, traj = _chunked_trajectory(im, x, self.micro_batch)
                mass = im.adapter.mass(im.model, x)

                batch_omega_dt: list[torch.Tensor] = []
                batch_eig_omega_dt: list[torch.Tensor] = []
                for ell in range(len(traj)):
                    h_ell = traj[ell].to(im.device)
                    ht = im.adapter.harmonic_terms(im.model, ell, x, h=h_ell)
                    if ht is None:
                        continue
                    k_diag, _s = ht
                    layers_seen.add(ell)
                    batch_omega_dt.append(
                        _omega_dt(k_diag, mass, dt).reshape(-1)
                    )

                    if has_weyl:
                        wp = im.adapter.well_parameters(
                            im.model, ell, x, h=h_ell
                        )
                        if wp is not None and wp["precision_lr"].shape[-1] > 0:
                            k_weyl = weyl_upper_bound(h_ell, wp)
                            batch_eig_omega_dt.append(
                                _omega_dt(k_weyl, mass, dt).reshape(-1)
                            )

                if batch_omega_dt:
                    omega_dt_blocks.append(torch.cat(batch_omega_dt))
                if batch_eig_omega_dt:
                    eig_omega_dt_blocks.append(torch.cat(batch_eig_omega_dt))

        if not omega_dt_blocks:
            return self._skip(
                "harmonic_terms() returned None for every sampled "
                "(layer, batch): this V_theta's harmonic_terms may not "
                "accept the shape this model's xi_module(h) produces"
            )

        omega_dt = torch.cat(omega_dt_blocks)
        stats = _quantiles(omega_dt)
        frac_marginal = _frac(omega_dt, self.marginal_bound)
        frac_unstable = _frac(omega_dt, self.stability_bound)
        ci = _bootstrap_frac_ci(
            omega_dt_blocks, self.stability_bound,
            n_boot=self.n_boot, seed=self.bootstrap_seed,
        )

        detail: dict = {
            "median": stats["median"],
            "p90": stats["p90"],
            "p99": stats["p99"],
            "p999": stats["p999"],
            "max": stats["max"],
            "frac_marginal": frac_marginal,
            "frac_unstable_ci95": ci,
            "n_layers": len(layers_seen),
            "n_samples": int(omega_dt.numel()),
            "stability_bound": self.stability_bound,
            "dt": dt,
        }

        if eig_omega_dt_blocks:
            eig_omega_dt = torch.cat(eig_omega_dt_blocks)
            eig_stats = _quantiles(eig_omega_dt)
            eig_frac_unstable = _frac(eig_omega_dt, self.stability_bound)
            eig_ci = _bootstrap_frac_ci(
                eig_omega_dt_blocks, self.stability_bound,
                n_boot=self.n_boot, seed=self.bootstrap_seed,
            )
            detail.update({
                "eig_median": eig_stats["median"],
                "eig_p90": eig_stats["p90"],
                "eig_p99": eig_stats["p99"],
                "eig_p999": eig_stats["p999"],
                "eig_max": eig_stats["max"],
                "eig_frac_unstable": eig_frac_unstable,
                "eig_frac_unstable_ci95": eig_ci,
            })

        return ProbeResult(
            name=self.name,
            statistic=frac_unstable,
            unit="frac_unstable",
            threshold=self.frac_unstable_threshold,
            passed=frac_unstable <= self.frac_unstable_threshold,
            detail=detail,
        )
