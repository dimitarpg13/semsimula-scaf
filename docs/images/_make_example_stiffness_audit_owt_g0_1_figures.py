#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Generate the REAL-data figures for
Example_Stiffness_Audit_OWT_g0.1_Anisotropic_Gaussian.md.

Unlike _make_stiffness_audit_figures.py (which is an explicitly labelled
SCHEMATIC), every number plotted by this script is copied verbatim from an
actual Phase 7 run of `scaf_checkpoint_analysis.ipynb` against four
checkpoints of the OWT aniso-Gaussian, gamma_train=0.10, Verlet-trained run
(`fock_aniso_owt_xi5long_topk16_dt32da16_mh4_aniso_dcvt5x8_ob_untied_wsd_e5c_
plgate_rep0.05_fockreg0.005_g0.1`, steps 9000/9500/10000/15000), plus the
15 real `watchdog_reload` steps read off that run's own `training_log.jsonl`.

Figure 1 (two panels, shared x-axis):
  - top:    omega*dt (p99, p99.9, max) at each audited checkpoint
  - bottom: val_ppl at the same four checkpoints
Both panels share the watchdog-reload vertical markers. The four audited
steps are plotted as discrete points connected by a thin dotted line only
to ease reading across the plot; the dotted segments do NOT represent
interpolated measurements -- there is a large, deliberately-unshaded gap
between step 10000 and step 15000 where no checkpoint was audited, and it
is exactly the window containing the densest cluster of reloads
(11550...13903).

Figure 2: a controlled synthetic verification (K=8 wells, d=16, rank=4,
random a_k/B_k/g_k, 500 trials) demonstrating that k_diag's per-dimension
max underestimates the true top eigenvalue of the effective precision
matrix in every trial, while the Weyl bound never does. This is NOT a
measurement from the checkpoint above -- Phase 7b has not been run against
it yet -- it verifies the *inequality itself*, independent of any specific
checkpoint's weights.

Run: python3 _make_example_stiffness_audit_owt_g0_1_figures.py
"""
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Real Phase 7 results (copied verbatim from the notebook's printed
#    output; see the markdown's results table for the source) ──────────
STEPS = [9000, 9500, 10000, 15000]
VAL_PPL = [194.67577502886286, 191.83942102914142, 184.1126151649017,
           192.5305992701286]
P99 = [0.1903, 0.1877, 0.1893, 0.1893]
P999 = [0.2166, 0.2119, 0.2181, 0.2148]
MAX_ = [0.6319, 0.3423, 0.4711, 0.4182]

# Real watchdog_reload steps from the same run's training_log.jsonl.
WATCHDOG_RELOAD_STEPS = [
    8925, 10409, 10697, 10721, 11550, 11898,
    12559, 12808, 13293, 13903, 15434, 15906, 16280, 16612, 16824,
]

fig, (ax1, ax2) = plt.subplots(
    2, 1, figsize=(9.5, 7.5), sharex=True,
    gridspec_kw={"height_ratios": [1.3, 1]},
)

ax1.plot(STEPS, P99, "o-", ms=6, lw=1.4, color="#3498db", label="p99")
ax1.plot(STEPS, P999, "s-", ms=6, lw=1.4, color="#8e44ad", label="p99.9")
ax1.plot(STEPS, MAX_, "^--", ms=7, lw=1.4, color="#e74c3c", label="max")
ax1.axhline(y=2.0, color="black", linewidth=1.6, linestyle=":",
            label="Verlet stability bound, omega dt = 2")
ax1.axhline(y=1.0, color="#7f8c8d", linewidth=1.2, linestyle="-.",
            label="marginal-instability threshold, omega dt = 1")
for i, r in enumerate(WATCHDOG_RELOAD_STEPS):
    ax1.axvline(x=r, color="#95a5a6", linewidth=1.0, alpha=0.6,
                label="watchdog reload (real)" if i == 0 else None)
ax1.axvspan(10000, 15000, color="#f1c40f", alpha=0.08)
ax1.annotate("no audited checkpoint in this window --\nincludes the densest reload cluster",
             xy=(12500, 0.55), ha="center", fontsize=8.5, color="#7f8c8d")
ax1.set_ylabel("omega * dt")
ax1.set_ylim(0, 2.3)
ax1.set_title(
    "REAL Phase 7 results -- OWT gamma=0.10, anisotropic Gaussian, Verlet-trained\n"
    "omega*dt (diagonal k_diag proxy) vs. training step"
)
ax1.legend(loc="upper left", fontsize=8.5, ncol=2)
ax1.grid(alpha=0.25)

ax2.plot(STEPS, VAL_PPL, "D-", ms=6.5, lw=1.6, color="#16a085")
for r in WATCHDOG_RELOAD_STEPS:
    ax2.axvline(x=r, color="#95a5a6", linewidth=1.0, alpha=0.6)
ax2.axvspan(10000, 15000, color="#f1c40f", alpha=0.08)
ax2.annotate("val_ppl backslides here\n(184.1 -> 192.5)",
             xy=(12500, 189), ha="center", fontsize=8.5, color="#c0392b")
ax2.set_xlabel("Training step")
ax2.set_ylabel("val_ppl")
ax2.grid(alpha=0.25)

fig.tight_layout()
out1 = "scaf_example_owt_g0_1_omega_dt_and_ppl_vs_step.png"
fig.savefig(out1, dpi=150)
print(f"Saved: {out1}")

# ── Figure 2: synthetic verification that k_diag's max can underestimate
#    the true top eigenvalue, and that the Weyl bound never does ──────
torch.manual_seed(1)
d, r, K, n_trials = 16, 4, 8, 500
true_maxes, diag_maxes, weyl_bounds = [], [], []
for _ in range(n_trials):
    a = torch.rand(K, d) * 2 + 0.01
    B = torch.randn(K, d, r) * 0.6
    g = torch.rand(K) * 0.5
    P_eff = torch.zeros(d, d)
    bound = 0.0
    for k in range(K):
        P_k = torch.diag(a[k]) + B[k] @ B[k].T
        P_eff += g[k] * P_k
        sigma_max_sq = torch.linalg.eigvalsh(B[k].T @ B[k]).max()
        bound += float(g[k] * (a[k].max() + sigma_max_sq))
    true_maxes.append(float(torch.linalg.eigvalsh(P_eff).max()))
    diag_maxes.append(float(torch.diagonal(P_eff).max()))
    weyl_bounds.append(bound)

true_maxes = np.array(true_maxes)
diag_maxes = np.array(diag_maxes)
weyl_bounds = np.array(weyl_bounds)
order = np.argsort(true_maxes)

fig2, ax = plt.subplots(figsize=(9, 5.2))
x = np.arange(n_trials)
ax.plot(x, weyl_bounds[order], ".", ms=3, color="#e67e22", alpha=0.6,
        label="Weyl upper bound (never below true max)")
ax.plot(x, true_maxes[order], "-", lw=1.8, color="#2c3e50",
        label="true top eigenvalue (exact eigh)")
ax.plot(x, diag_maxes[order], ".", ms=3, color="#3498db", alpha=0.6,
        label="k_diag's max (diagonal proxy, underestimates every trial)")
ax.set_xlabel(f"trial (sorted by true top eigenvalue, n={n_trials})")
ax.set_ylabel("effective curvature")
ax.set_title(
    "Synthetic verification, K=8 wells / d=16 / rank=4 (NOT this checkpoint's "
    "real wells)\nk_diag's max always underestimates; the Weyl bound never does"
)
ax.legend(loc="upper left", fontsize=9)
ax.grid(alpha=0.25)
fig2.tight_layout()
out2 = "scaf_example_diag_blind_spot_synthetic_verification.png"
fig2.savefig(out2, dpi=150)
print(f"Saved: {out2}")

underestimate_rate = float((diag_maxes < true_maxes).mean())
mean_gap = float((true_maxes - diag_maxes).mean())
print(f"k_diag underestimated the true top eigenvalue in "
      f"{underestimate_rate:.0%} of {n_trials} trials "
      f"(mean gap {mean_gap:.3f})")
