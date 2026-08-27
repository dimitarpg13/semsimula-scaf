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
measurement from the checkpoint above -- it verifies the *inequality
itself*, independent of any specific checkpoint's weights.

Figure 3: the REAL Phase 7b result against the same four checkpoints --
diagonal-proxy `max`/`frac(>2)` side by side with the Weyl-bound
`eig_max`/`eig_frac(>2)`, copied verbatim from the notebook's printed
output. Unlike Figure 2, this IS a measurement from this checkpoint's own
trained wells, not a synthetic sanity check.

Figure 4: why there is no checkpoint to audit inside (10000, 15000) at
all -- the FULL-resolution val_ppl and running best_ppl series read
directly off training_log.jsonl's own EVAL_INTERVAL=500 rows (not
resampled or smoothed), together with every real watchdog_reload step.
Unlike Figures 1-3, this needs no model weights at all -- it is pure
log analysis -- and it is exactly why no additional checkpoint exists to
fill that gap after the fact: best_ppl is set at step 10000 (184.11) and
never improves again anywhere in the logged run (through step 17000,
where the run was manually stopped), so the ONE trigger
(`val_ppl < best_val_ppl`) that would have produced a `_best.pt` save
inside that window simply never fired.

Figure 5: the REAL Phase 7c (native=True) cross-check against the same
four checkpoints -- Weyl-bound `max`/`frac(>2)` under the forced
`baoab_cfc` trajectory (Phase 7b, Figure 3) side by side with the same
statistic measured along each checkpoint's own, unmodified Verlet
trajectory. This settles the trajectory-substitution caveat raised
alongside Figure 3: `frac(>2)` agrees to within 0.006 percentage points
at every checkpoint, ~150x smaller than the 0.924-point trend across the
run, while `max` (a single most-extreme sampled position) moves more
between modes, as expected for a statistic that individual trajectory
divergence is free to affect.

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

# ── Figure 3: REAL Phase 7b results against this checkpoint's own wells ──
EIG_DIAG_MAX = MAX_                                   # same as Figure 1
EIG_WEYL_MAX = [4.2452, 2.7128, 2.9264, 2.6572]
EIG_DIAG_FRAC = [0.0, 0.0, 0.0, 0.0]
EIG_WEYL_FRAC = [5.282e-02, 5.619e-02, 5.932e-02, 6.206e-02]

fig3, (ax3, ax4) = plt.subplots(
    2, 1, figsize=(9.5, 7.5), sharex=True,
    gridspec_kw={"height_ratios": [1.2, 1]},
)

ax3.plot(STEPS, EIG_DIAG_MAX, "o-", ms=6.5, lw=1.6, color="#3498db",
         label="max, diagonal k_diag proxy (Phase 7)")
ax3.plot(STEPS, EIG_WEYL_MAX, "^--", ms=7.5, lw=1.6, color="#e67e22",
         label="max, Weyl upper bound (Phase 7b)")
ax3.axhline(y=2.0, color="black", linewidth=1.6, linestyle=":",
            label="Verlet stability bound, omega dt = 2")
for i, r in enumerate(WATCHDOG_RELOAD_STEPS):
    ax3.axvline(x=r, color="#95a5a6", linewidth=1.0, alpha=0.6,
                label="watchdog reload (real)" if i == 0 else None)
ax3.set_ylabel("omega * dt (max over sampled tokens)")
ax3.set_title(
    "REAL Phase 7b result -- OWT gamma=0.10, anisotropic Gaussian, Verlet-trained\n"
    "diagonal proxy vs. Weyl upper bound, at this checkpoint's own wells"
)
ax3.legend(loc="upper left", fontsize=8.5)
ax3.grid(alpha=0.25)

ax4.plot(STEPS, [f * 100 for f in EIG_DIAG_FRAC], "o-", ms=6.5, lw=1.6,
         color="#3498db", label="frac(omega*dt>2), diagonal proxy")
ax4.plot(STEPS, [f * 100 for f in EIG_WEYL_FRAC], "^--", ms=7.5, lw=1.6,
         color="#e67e22", label="frac(omega*dt>2), Weyl upper bound")
for r in WATCHDOG_RELOAD_STEPS:
    ax4.axvline(x=r, color="#95a5a6", linewidth=1.0, alpha=0.6)
ax4.annotate("diagonal proxy: exactly 0% at all four steps",
             xy=(11500, 0.3), ha="center", fontsize=8.5, color="#2980b9")
ax4.set_xlabel("Training step")
ax4.set_ylabel("% of sampled (token, layer) positions\nwith omega*dt > 2")
ax4.grid(alpha=0.25)
ax4.legend(loc="center left", fontsize=8.5)

fig3.tight_layout()
out3 = "scaf_example_owt_g0_1_phase7b_real_diag_vs_weyl.png"
fig3.savefig(out3, dpi=150)
print(f"Saved: {out3}")

# ── Figure 4: full-resolution val_ppl/best_ppl vs step, straight from
#    training_log.jsonl -- shows WHY there is no checkpoint to audit
#    inside (10000, 15000): best_ppl never moves again after step 10000
#    anywhere in the logged run. Every (step, val_ppl, best_ppl) triple
#    below is copied verbatim from that log's 'val_ppl' rows for
#    9000 <= step <= 17000 (duplicate step numbers are real -- the run
#    was resumed at least twice in this window, itself corroborating
#    how disruptive this stretch of training was) ────────────────────
LOG_STEPS = [9000, 9500, 10000, 10500, 11000, 11500, 10500, 11000, 11500,
             12000, 12500, 13000, 13500, 14000, 14500, 15000, 15500,
             16000, 16500, 17000]
LOG_VAL_PPL = [194.68, 191.84, 184.11, 197.48, 192.54, 189.11, 191.61,
               191.15, 195.52, 185.27, 190.15, 191.89, 186.83, 194.76,
               189.88, 192.53, 186.78, 186.98, 193.63, 190.47]
LOG_BEST_PPL = [184.11 if s >= 10000 else v
                for s, v in zip(LOG_STEPS, LOG_VAL_PPL)]
# best_ppl is non-decreasing-in-reverse by construction (a running min);
# the line above is exact for this window because step 10000 IS where
# the last real improvement happened -- verified directly against the
# log's own 'best_ppl' field, not re-derived.

fig4, ax5 = plt.subplots(figsize=(9.5, 5.2))
order4 = np.argsort(LOG_STEPS)
steps4 = np.array(LOG_STEPS)[order4]
ppl4 = np.array(LOG_VAL_PPL)[order4]
ax5.plot(steps4, ppl4, "o", ms=5, color="#16a085", alpha=0.85,
         label="val_ppl (every EVAL_INTERVAL=500 step, incl. resumes)")
ax5.plot([9000, 10000], [194.68, 184.11], "-", lw=1.2, color="#16a085",
         alpha=0.5)
ax5.axhline(y=184.11, color="#c0392b", linewidth=1.8, linestyle="--",
            label="best_ppl (frozen from step 10000 onward)")
for i, r in enumerate(WATCHDOG_RELOAD_STEPS):
    if 9000 <= r <= 17000:
        ax5.axvline(x=r, color="#95a5a6", linewidth=1.0, alpha=0.6,
                    label="watchdog reload (real)" if i == 1 else None)
ax5.axvspan(10000, 17000, color="#e74c3c", alpha=0.05)
ax5.set_ylim(183, 202)
ax5.annotate(
    "best_ppl never improves again in the entire logged run\n"
    "(step 10,000 -> 17,000, the run's manual stop point) --\n"
    "the ONLY save trigger that could have produced a\n"
    "checkpoint here never fires",
    xy=(13800, 199.5), ha="center", va="top", fontsize=8.5, color="#c0392b")
ax5.set_xlabel("Training step")
ax5.set_ylabel("val_ppl")
ax5.set_title(
    "Why (10000, 15000) has no checkpoint to audit -- full-resolution "
    "val_ppl vs. best_ppl\nOWT gamma=0.10, anisotropic Gaussian, "
    "Verlet-trained (log data only, no weights needed)"
)
ax5.legend(loc="lower right", fontsize=8.5)
ax5.grid(alpha=0.25)
fig4.tight_layout()
out4 = "scaf_example_owt_g0_1_best_ppl_frozen_no_checkpoint_gap.png"
fig4.savefig(out4, dpi=150)
print(f"Saved: {out4}")

# ── Figure 5: REAL Phase 7c (native=True) cross-check -- Weyl bound,
#    forced (baoab_cfc) vs. native (Verlet) trajectory, same four
#    checkpoints. Copied verbatim from the notebook's printed output ──
NATIVE_WEYL_MAX = [3.0195, 4.0838, 2.7457, 3.1421]
NATIVE_WEYL_FRAC = [5.276e-02, 5.621e-02, 5.932e-02, 6.204e-02]

fig5, (ax6, ax7) = plt.subplots(
    2, 1, figsize=(9.5, 7.5), sharex=True,
    gridspec_kw={"height_ratios": [1.2, 1]},
)
ax6.plot(STEPS, EIG_WEYL_MAX, "o-", ms=6.5, lw=1.6, color="#3498db",
         label="Weyl max, forced baoab_cfc trajectory (Phase 7b)")
ax6.plot(STEPS, NATIVE_WEYL_MAX, "^--", ms=7.5, lw=1.6, color="#8e44ad",
         label="Weyl max, native Verlet trajectory (Phase 7c)")
ax6.axhline(y=2.0, color="black", linewidth=1.6, linestyle=":",
            label="Verlet stability bound, omega dt = 2")
for i, r in enumerate(WATCHDOG_RELOAD_STEPS):
    ax6.axvline(x=r, color="#95a5a6", linewidth=1.0, alpha=0.6,
                label="watchdog reload (real)" if i == 0 else None)
ax6.set_ylabel("omega * dt (Weyl bound, max)")
ax6.set_title(
    "REAL Phase 7c cross-check -- OWT gamma=0.10, anisotropic Gaussian\n"
    "Weyl bound: forced trajectory vs. this checkpoint's own native Verlet trajectory"
)
ax6.legend(loc="upper left", fontsize=8.5)
ax6.grid(alpha=0.25)

ax7.plot(STEPS, [f * 100 for f in EIG_WEYL_FRAC], "o-", ms=6.5, lw=1.6,
         color="#3498db", label="frac(omega*dt>2), forced")
ax7.plot(STEPS, [f * 100 for f in NATIVE_WEYL_FRAC], "^--", ms=7.5, lw=1.6,
         color="#8e44ad", label="frac(omega*dt>2), native")
for r in WATCHDOG_RELOAD_STEPS:
    ax7.axvline(x=r, color="#95a5a6", linewidth=1.0, alpha=0.6)
ax7.annotate(
    "agree to within 0.006pp at every step\n"
    "(~150x smaller than the 0.92pp trend)",
    xy=(11500, 5.5), ha="center", fontsize=8.5, color="#27ae60")
ax7.set_xlabel("Training step")
ax7.set_ylabel("% of sampled (token, layer) positions\nwith omega*dt > 2")
ax7.grid(alpha=0.25)
ax7.legend(loc="upper left", fontsize=8.5)

fig5.tight_layout()
out5 = "scaf_example_owt_g0_1_phase7c_native_vs_forced.png"
fig5.savefig(out5, dpi=150)
print(f"Saved: {out5}")
