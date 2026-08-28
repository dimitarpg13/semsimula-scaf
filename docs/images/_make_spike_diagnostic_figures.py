#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Generate the figures for the gradient-spike diagnostic work.

Three figures, used by:
  - semsimula-scaf/docs/Gradient_Spike_Probe_Requirements_and_Design.md
  - semsimula-paper/companion_notes/CfC_BAOAB_Integrator_and_Mitigations.md (§33)

Figure 1 (scaf_spike_diag_sigma_lr_bracket_result.png)
    REAL DATA. The measured sigma_max(B_k)^2 percentiles from the live L=8
    baoab_cfc d=384 run: a healthy checkpoint (step 27,000, best, PPL 100.47)
    vs. a spike-regime prereload snapshot (step 34,091, hard-watchdog trigger).
    The two distributions are within +8% at every percentile -> B_k growth is
    NOT the driver of these bursts (companion note §31.4 step 1, §33).

Figure 2 (scaf_spike_diag_forward_backward_map.png)
    SCHEMATIC. The Fock-PARFLM forward integrator, the parameter groups that
    spike, the numerically-risky op inside each, their per-group clip ceilings,
    and the second-order backward grad flow the probe instruments. Marks the
    two groups excluded from the watchdog aggregate.

Figure 3 (scaf_spike_diag_probe_pipeline.png)
    SCHEMATIC. The GradientSpikeProbe data flow: pinned (weights, offending
    batch) -> one isolated forward+backward under the save/zero/restore .grad
    invariant -> per-group/per-parameter/per-layer grad-norm quantiles +
    forward-activation extremes -> attribution verdict, branching to the
    right remediation lever.

Run: python3 _make_spike_diagnostic_figures.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# Shared palette (matches the existing scaf_* figures).
C_BLUE = "#3498db"
C_RED = "#e74c3c"
C_GREEN = "#27ae60"
C_ORANGE = "#e67e22"
C_GREY = "#7f8c8d"
C_PURPLE = "#8e44ad"
C_DARK = "#2c3e50"


# ---------------------------------------------------------------------------
# Figure 1 — the sigma_lr bracketing result (REAL DATA)
# ---------------------------------------------------------------------------
def fig_bracket_result(out="scaf_spike_diag_sigma_lr_bracket_result.png"):
    labels = ["p50", "p90", "p99", "p99.9", "max"]
    healthy = [282.11, 663.61, 1047.00, 2322.33, 6364.81]
    spike = [305.59, 700.53, 1082.80, 2499.07, 6427.16]
    deltas = [100.0 * (s - h) / h for h, s in zip(healthy, spike)]

    x = np.arange(len(labels))
    w = 0.38

    fig, ax = plt.subplots(figsize=(9.6, 5.4))
    b1 = ax.bar(x - w / 2, healthy, w, color=C_BLUE,
                label="healthy — step 27,000 best (PPL 100.47)")
    b2 = ax.bar(x + w / 2, spike, w, color=C_RED,
                label="spike-regime — step 34,091 prereload (hard trigger)")

    ax.set_yscale("log")
    ax.set_ylabel(r"$\sigma_{\max}(B_k)^2$  (per well, per xi-channel, per layer)")
    ax.set_xlabel("percentile of the pooled distribution (1,310,720 samples each)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title(
        "The low-rank curvature barely moves into the crisis\n"
        "L=8  baoab_cfc  d=384  aniso-Gaussian $V_\\theta$  (fixed seed-0 probe batch)"
    )
    ax.set_ylim(100, 12000)

    for xi, h, s, d in zip(x, healthy, spike, deltas):
        top = max(h, s)
        ax.annotate(f"+{d:.1f}%", xy=(xi, top), xytext=(0, 8),
                    textcoords="offset points", ha="center", va="bottom",
                    fontsize=9, color=C_DARK, fontweight="bold")

    ax.text(
        0.015, 0.97,
        "All percentiles within +8% -> $B_k$ growth is NOT the driver\n"
        "of these bursts (companion note §31.4 step 1, §33).\n"
        "precision_lr_max is the wrong lever; look at the\n"
        "non-$V_\\theta$ groups instead.",
        transform=ax.transAxes, ha="left", va="top", fontsize=9.5,
        bbox=dict(boxstyle="round,pad=0.5", fc="#fdf6e3", ec=C_ORANGE, lw=1.3),
    )
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(axis="y", alpha=0.25, which="both")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# small helpers for the schematic figures
# ---------------------------------------------------------------------------
def _box(ax, xy, w, h, text, fc, ec=C_DARK, fontsize=9.5, tc="white", lw=1.4):
    x, y = xy
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.015,rounding_size=0.04",
        fc=fc, ec=ec, lw=lw, mutation_aspect=1.0))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize, color=tc, zorder=5)


def _arrow(ax, p0, p1, color=C_DARK, lw=1.8, style="-|>", ls="-", rad=0.0):
    ax.add_patch(FancyArrowPatch(
        p0, p1, arrowstyle=style, mutation_scale=14, lw=lw, color=color,
        linestyle=ls, connectionstyle=f"arc3,rad={rad}", zorder=4))


# ---------------------------------------------------------------------------
# Figure 2 — forward/backward map of the spiking groups (SCHEMATIC)
# ---------------------------------------------------------------------------
def fig_forward_backward_map(out="scaf_spike_diag_forward_backward_map.png"):
    fig, ax = plt.subplots(figsize=(12.0, 8.2))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 11)
    ax.axis("off")

    cx, bw, bh = 3.55, 3.3, 0.74  # center column boxes

    # forward-pass stages, top -> bottom  (y = box bottom)
    stages = [
        (9.0, "E(x) + P[pos]  ->  h0", C_DARK),
        (7.6, "creation_gate (QKV softmax readout)", C_ORANGE),
        (6.2, "PARF dynamics:  V_theta + V_phi", C_BLUE),
        (4.8, "reverse channel (non-conservative kick)", C_PURPLE),
        (3.4, "destruction_gate (register decay)", C_ORANGE),
        (1.7, "h_L  ->  logits  ->  cross-entropy loss", C_DARK),
    ]
    for y, txt, c in stages:
        _box(ax, (cx, y), bw, bh, txt, fc=c, fontsize=8.2)

    # the L-layer repeat bracket (creation ... destruction)
    y_lo, y_hi = 3.4, 7.6 + bh
    ax.annotate("", xy=(cx - 0.30, y_lo), xytext=(cx - 0.30, y_hi),
                arrowprops=dict(arrowstyle="-", color=C_GREY, lw=1.2))
    ax.text(cx - 0.52, (y_lo + y_hi) / 2, "x L layers",
            rotation=90, ha="center", va="center", fontsize=8.5, color=C_GREY)

    # forward arrows
    ys = [s[0] for s in stages]
    for y_top, y_bot in zip(ys[:-1], ys[1:]):
        _arrow(ax, (cx + bw / 2, y_top), (cx + bw / 2, y_bot + bh),
               color=C_GREY, lw=1.4)

    # backward (second-order) grad flow, far left, going up
    _arrow(ax, (1.55, 2.0), (1.55, 9.3), color=C_RED, lw=2.6, style="-|>")
    ax.text(1.28, 5.6, "backward grad flow\n(2nd-order chain,\ncreate_graph=True)",
            rotation=90, ha="center", va="center", fontsize=8.2, color=C_RED)

    # right-hand annotations: risky op + clip ceiling per spiking group
    rbh = 1.16
    ann = [
        (9.0, "E / P", "rank-1 CE-through-softmax grad;", "no LN before the residual add", "clip 0.3*"),
        (7.6, "creation_gate", "sharp softmax: tau=exp(log_tau).clamp(1e-4);", "cumsum / Z.clamp(1e-30)", "clip 0.3"),
        (6.2, "depth_code", "shifts xi into stiffer wells;", "boundary layers dominate (§27)", "clip 0.25"),
        (4.8, "reverse_channel_scale + reverse_ch", "(dt^2/m).tanh(s).Q_force injection;", "softmax readout saturation", "clip 0.1  [watchdog-excluded]"),
        (3.4, "register", "hard salience gate (thr 0.005);", "near-discontinuous grad", "clip 0.3"),
    ]
    ax_r, rw = 7.15, 4.6
    for y, grp, risk1, risk2, clip in ann:
        by = y + bh / 2 - rbh / 2  # vertically center the annotation on the stage
        _box(ax, (ax_r, by), rw, rbh, "", fc="#f7f9fb", ec=C_GREY, lw=1.0)
        ax.text(ax_r + 0.14, by + rbh - 0.16, grp, ha="left", va="top",
                fontsize=8.6, color=C_DARK, fontweight="bold")
        ax.text(ax_r + 0.14, by + rbh - 0.52, risk1, ha="left", va="top",
                fontsize=7.4, color="#555555")
        ax.text(ax_r + 0.14, by + rbh - 0.82, risk2, ha="left", va="top",
                fontsize=7.4, color="#555555")
        clip_col = C_RED if "excluded" in clip else C_GREEN
        ax.text(ax_r + rw - 0.12, by + 0.12, clip, ha="right", va="bottom",
                fontsize=7.4, color=clip_col, fontweight="bold")
        _arrow(ax, (cx + bw, y + bh / 2), (ax_r, by + rbh / 2),
               color=C_GREY, lw=1.0, style="-|>", rad=0.0)

    ax.text(0.12, 10.5,
            "Where the L=8 bursts live: the non-$V_\\theta$ groups",
            ha="left", va="center", fontsize=13, color=C_DARK,
            fontweight="bold")
    ax.text(0.12, 10.05,
            "=P and =E exact-match clip added at d=1024 (§23.3). "
            "SCHEMATIC — arrows show structure, not measured magnitudes.",
            ha="left", va="center", fontsize=8.2, color=C_GREY)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# Figure 3 — the GradientSpikeProbe pipeline (SCHEMATIC)
# ---------------------------------------------------------------------------
def fig_probe_pipeline(out="scaf_spike_diag_probe_pipeline.png"):
    fig, ax = plt.subplots(figsize=(11.6, 6.6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.axis("off")

    # inputs (left)
    _box(ax, (0.3, 6.2), 3.2, 0.8, "checkpoint weights\n(healthy  |  spike prereload)",
         fc=C_BLUE, fontsize=8.6)
    _box(ax, (0.3, 4.9), 3.2, 0.8, "captured offending batch\n(x, y, RNG)  — Phase 1",
         fc=C_GREEN, fontsize=8.6)

    # core (center)
    _box(ax, (4.3, 5.05), 3.4, 1.5,
         "isolated forward + backward\n\nsave -> zero -> restore .grad\n"
         "(monitor non-pollution invariant)",
         fc=C_DARK, fontsize=8.8)

    # outputs (right)
    _box(ax, (8.4, 6.35), 3.3, 0.95,
         "per-group / per-parameter\n/ per-layer grad-norm quantiles",
         fc=C_ORANGE, fontsize=8.3)
    _box(ax, (8.4, 5.15), 3.3, 0.95,
         "forward-activation extremes\n(softmax logits, tau, Q_force, ...)",
         fc=C_ORANGE, fontsize=8.3)

    # verdict + branches (bottom)
    _box(ax, (4.3, 2.7), 3.4, 0.95,
         "attribution verdict\ngroup x layer x op",
         fc=C_PURPLE, fontsize=9.0)
    _box(ax, (0.3, 1.0), 4.0, 1.0,
         "V_theta / B_k implicated?\n-> bracket sigma_lr -> precision_lr_max\n(companion §31.4, §29.3)",
         fc="#f7f9fb", ec=C_GREY, tc=C_DARK, fontsize=8.0)
    _box(ax, (7.5, 1.0), 4.0, 1.0,
         "non-$V_\\theta$ group implicated?\n-> targeted per-group clip / op fix\n(companion §33)",
         fc="#fdf6e3", ec=C_ORANGE, tc=C_DARK, fontsize=8.0)

    # arrows
    _arrow(ax, (3.5, 6.6), (4.3, 6.1), color=C_GREY)
    _arrow(ax, (3.5, 5.3), (4.3, 5.7), color=C_GREY)
    _arrow(ax, (7.7, 6.1), (8.4, 6.8), color=C_GREY)
    _arrow(ax, (7.7, 5.6), (8.4, 5.6), color=C_GREY)
    _arrow(ax, (6.0, 5.05), (6.0, 3.65), color=C_PURPLE, lw=2.0)
    _arrow(ax, (4.6, 2.7), (2.3, 2.0), color=C_GREY, style="-|>", rad=-0.15)
    _arrow(ax, (7.4, 2.7), (9.5, 2.0), color=C_GREY, style="-|>", rad=0.15)

    ax.text(0.12, 7.6, "GradientSpikeProbe: from a captured burst to a lever",
            ha="left", va="center", fontsize=13, color=C_DARK,
            fontweight="bold")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    fig_bracket_result()
    fig_forward_backward_map()
    fig_probe_pipeline()
