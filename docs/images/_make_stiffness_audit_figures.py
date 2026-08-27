#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Generate the schematic omega*dt-vs-step trend figure for
Stiffness_Audit_Requirements_and_Design.md.

This is a SCHEMATIC illustration, not a plot of measured data. Only one
ingredient is real: the vertical dashed lines mark the exact watchdog-reload
steps pulled from the ground-truth training log of the OWT
aniso-Gaussian, gamma_train=0.10, Verlet run (see
`~/Downloads/semsimula_fock_aniso_gaussian_fockreg_owt_xi5long_topk16_dt32da16_
mh4_aniso_dcvt5x8_ob_untied_wsd_e5c_plgate_rep0.05_fockreg0.005_g0.1/results/
training_log.jsonl`, event == "watchdog_reload"). The omega*dt curves
themselves are synthetic, hand-picked only to look like what a run with that
reload pattern would plausibly produce: a rising p99 / max and a growing
frac_unstable, without ever having actually run Phase 7 against this run's
checkpoints. Do not read numeric values off this figure as measurements.

Run: python3 _make_stiffness_audit_figures.py
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Ground truth: watchdog_reload steps from the real training_log.jsonl for
# the OWT aniso-Gaussian gamma_train=0.10 Verlet run (15 reloads total).
WATCHDOG_RELOAD_STEPS = [
    8925, 10409, 10697, 10721, 11550, 11898,
    12559, 12808, 13293, 13903, 15434, 15906, 16280, 16612, 16824,
]

rng = np.random.default_rng(0)
steps = np.arange(0, 17500, 250)

# Synthetic p99 / max omega*dt trend: flat and comfortably sub-bound early,
# then a rising trend with reload-correlated bumps, purely for illustration.
base_p99 = 0.3 + 0.9 * (steps / steps.max()) ** 1.6
base_max = 0.8 + 3.4 * (steps / steps.max()) ** 1.8
bump = np.zeros_like(steps, dtype=float)
for r in WATCHDOG_RELOAD_STEPS:
    bump += 1.1 * np.exp(-0.5 * ((steps - r) / 220.0) ** 2)
p99 = base_p99 + 0.15 * bump + rng.normal(0, 0.03, size=steps.shape)
omega_dt_max = base_max + 0.9 * bump + rng.normal(0, 0.08, size=steps.shape)
p99 = np.clip(p99, 0.05, None)
omega_dt_max = np.clip(omega_dt_max, 0.05, None)

fig, ax = plt.subplots(figsize=(9.5, 5.2))
ax.plot(steps, p99, "o-", ms=3.5, lw=1.6, color="#3498db", label="p99 (schematic)")
ax.plot(steps, omega_dt_max, "s--", ms=3.5, lw=1.6, color="#e74c3c",
        label="max (schematic)")
ax.axhline(y=2.0, color="black", linewidth=1.6, linestyle=":",
           label="Verlet stability bound, omega dt = 2")

for i, r in enumerate(WATCHDOG_RELOAD_STEPS):
    ax.axvline(x=r, color="#7f8c8d", linewidth=1.0, alpha=0.55,
               label="watchdog reload (real steps)" if i == 0 else None)

ax.set_xlabel("Training step")
ax.set_ylabel("omega * dt")
ax.set_title(
    "SCHEMATIC — omega*dt vs. step\n"
    "(curves are illustrative; grey lines are the real gamma=0.10 reload steps)"
)
ax.set_ylim(0, 6.5)
ax.legend(loc="upper left", fontsize=9)
ax.grid(alpha=0.25)
fig.tight_layout()

out_path = "scaf_stiffness_omega_dt_trend_schematic.png"
fig.savefig(out_path, dpi=150)
print(f"Saved: {out_path}")
