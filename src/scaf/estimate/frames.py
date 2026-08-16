"""Tidy interventional frames — the bridge from a probe to a formal estimand.

A probe answers "is there a leak, and how big?". A frame turns the same
measurement into a **dataset**, so the leak can be handed to a causal-inference
engine and interrogated with the standard verbs: identify, estimate, refute,
and heterogeneity.

The dataset is *interventional by construction*. Each row is one scored target
position under one arm:

* ``future_perturbed = 0`` — the factual run, the true suffix in place.
* ``future_perturbed = 1`` — the same prefix under
  :math:`\\mathrm{do}(x_{>t_p} := \\tilde x_{>t_p})`.

Because we assign the arm ourselves, treatment is exogenous: nothing causes it,
so no backdoor set is needed and identification is free. That is the structural
difference between auditing a simulator and analysing observational data, and
it is why the naive difference in means is already the ATE. What DoWhy and
EconML add on top is *refutation* and *heterogeneity*, not identification —
see :mod:`scaf.estimate.pywhy`.

**Outcome.** The outcome is the negative log-likelihood of the true next token,
so the ATE is :math:`\\Delta\\mathrm{NLL}` in nats and is directly comparable to
the honest-vs-standard perplexity gap that
:class:`~scaf.probes.target_relocation.TargetRelocationProbe` reports. Logit
space is available as a secondary outcome, but nats are the unit in which a
leak's cost is actually denominated.

**Targets are held fixed across arms.** The true next token always comes from
the factual sequence. The question is how the model's belief about *that* token
moves when the future changes — scoring against the resampled token instead
would measure something else entirely.

**One position is special.** At :math:`t = t_p` the target :math:`x_{t_p+1}` is
itself the first resampled token, so this row measures "can the model see the
very next token", which is the sharpest form of the leak. Every other row
measures "does the distant future move a prediction whose target did not
change". These are different questions, so the frame tags them with
``target_perturbed`` and keeps them separable rather than averaging a sharp
effect into a diffuse one.

**Pairing.** Every counterfactual row is emitted alongside the factual row it
is compared against, sharing a ``unit_id``. The factual measurement is
deterministic — the same input yields the same logits — so re-emitting it per
pair is not fabricated data, it is the same reference value reused. Pairing
makes :meth:`LeakFrame.ate` an exact mean paired difference rather than a
difference of group means over unbalanced strata.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F

from ..core.corpus import Corpus, SyntheticCorpus, TokenCorpus
from ..core.intervenable import InterventableModel
from ..probes.future_perturbation import (
    _chunked_logits,
    make_perturbation_pairs,
)
from ..probes.basin_membership import assign_dominant_wells
from ..probes.hidden_state import _chunked_trajectory, _cosine_deviation

__all__ = ["LeakFrame", "build_leak_frame"]


def _mean(xs) -> float:
    xs = list(xs)
    return sum(xs) / len(xs) if xs else 0.0


def _sort_key(kv):
    """Order strata numerically when they are numbers, else as text.

    ``distance_to_cut`` is the axis a reader scans for a decay profile, and
    sorting it lexicographically would interleave 2 between 17 and 20.
    """
    k = kv[0]
    return (0, float(k), "") if isinstance(k, (int, float)) else (1, 0.0, str(k))


@dataclass
class LeakFrame:
    """A tidy interventional dataset ready for causal estimation.

    Column-oriented so it converts to a DataFrame without a transpose, and so
    the pandas dependency stays optional: every statistic this class computes
    itself is plain Python.

    Attributes:
        columns: Column name to list of values. All lists share a length.
        treatment: Name of the binary treatment column.
        outcome: Name of the default outcome column.
        effect_modifiers: Columns along which the effect may vary — the
            candidate axes for a CATE analysis.
        common_causes: Backdoor set. Empty, and that is the point: treatment
            was randomised by the auditor, so there is nothing to adjust for.
        metadata: Provenance — model, adapter, dtype, corpus shape, splits.
    """

    columns: dict[str, list] = field(default_factory=dict)
    treatment: str = "future_perturbed"
    outcome: str = "nll"
    effect_modifiers: tuple[str, ...] = (
        "position",
        "split_frac",
        "target_perturbed",
        "distance_to_cut",
    )
    common_causes: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(next(iter(self.columns.values()), []))

    @property
    def n_units(self) -> int:
        """Number of factual/counterfactual pairs."""
        return len(set(self.columns.get("unit_id", [])))

    def column(self, name: str) -> list:
        if name not in self.columns:
            raise KeyError(
                f"no column {name!r}; available: {sorted(self.columns)}"
            )
        return self.columns[name]

    # ------------------------------------------------------------------
    # Estimation without any extra dependency
    # ------------------------------------------------------------------
    def deltas(self, outcome: str | None = None) -> list[float]:
        """Per-pair treated-minus-control differences.

        The frame's sufficient statistic: everything about the effect is in
        these numbers, and the pairing that produced them is what makes the
        inference in :meth:`ate_test` exact.
        """
        y = self.column(outcome or self.outcome)
        t = self.column(self.treatment)
        units = self.column("unit_id")
        treated: dict[Any, float] = {}
        control: dict[Any, float] = {}
        for yi, ti, ui in zip(y, t, units, strict=True):
            (treated if ti else control)[ui] = yi
        return [
            treated[u] - control[u] for u in treated.keys() & control.keys()
        ]

    def ate(self, outcome: str | None = None) -> float:
        """Exact mean paired difference — the ATE of ``do(future)``.

        Treatment is randomised and every counterfactual row is paired with its
        factual reference, so this *is* the causal effect, not an approximation
        of it. :mod:`scaf.estimate.pywhy` should reproduce this number; a
        disagreement means the estimator is misconfigured, not that it found
        something subtler.
        """
        return _mean(self.deltas(outcome))

    def ate_test(
        self,
        outcome: str | None = None,
        *,
        n_permutations: int = 10_000,
        n_bootstrap: int = 2_000,
        alpha: float = 0.05,
        seed: int = 0,
        chunk: int = 256,
    ) -> dict[str, Any]:
        """Exact paired significance test and a paired bootstrap interval.

        Under the null that the future cannot reach the past, the two arms of a
        pair are exchangeable, so flipping the sign of any pair's difference
        leaves the distribution unchanged. Enumerating random sign flips
        therefore gives an *exact* p-value — no normality assumption, no
        homoscedasticity assumption, and no bootstrap of the whole dataset.

        Both assumptions matter here, which is why this test exists rather than
        deferring to a generic one. Leak effects are wildly heteroscedastic:
        the position adjacent to the cut can carry a hundred nats while every
        other position carries exactly zero. A test that pools rows and
        permutes the treatment label across the whole frame destroys the
        pairing that carries the entire signal, and can return a
        non-significant p-value for an effect its own confidence interval puts
        nowhere near zero. That is a false all-clear, which is the one failure
        mode this library exists to prevent.

        Args:
            outcome: Outcome column; defaults to the frame's.
            n_permutations: Sign-flip draws. The smallest reportable p-value is
                ``1 / (n_permutations + 1)``.
            n_bootstrap: Resamples for the confidence interval. Zero disables.
            alpha: Interval level; ``0.05`` gives a 95% interval.
            seed: RNG seed, for a reproducible p-value.
            chunk: Permutations evaluated per batch, to bound peak memory.

        Returns:
            ``{"ate", "p_value", "ci", "stderr", "n_units", "n_permutations"}``.
        """
        d = torch.tensor(self.deltas(outcome), dtype=torch.float64)
        n = d.numel()
        out: dict[str, Any] = {
            "ate": float(d.mean()) if n else 0.0,
            "n_units": int(n),
            "n_permutations": int(n_permutations),
            "p_value": None,
            "ci": None,
            "stderr": None,
        }
        if n == 0:
            return out
        out["stderr"] = float(d.std(unbiased=True) / (n ** 0.5)) if n > 1 else 0.0

        obs = abs(float(d.mean()))
        gen = torch.Generator().manual_seed(seed)

        if obs == 0.0 and float(d.abs().max()) == 0.0:
            # Every pair is bit-identical, so no sign assignment can produce a
            # non-zero mean. The permutation p-value is 1 by construction;
            # computing it would waste a second to rediscover that.
            out.update(p_value=1.0, ci=(0.0, 0.0))
            return out

        exceed = 0
        done = 0
        while done < n_permutations:
            m = min(chunk, n_permutations - done)
            signs = (
                torch.randint(0, 2, (m, n), generator=gen, dtype=torch.float64)
                * 2.0 - 1.0
            )
            exceed += int(
                ((signs * d).mean(dim=1).abs() >= obs - 1e-15).sum()
            )
            done += m
        # The +1 correction keeps the p-value from ever being exactly zero: we
        # have finite resolution and should not claim more than we sampled.
        out["p_value"] = (exceed + 1) / (n_permutations + 1)

        if n_bootstrap > 0:
            means = []
            done = 0
            while done < n_bootstrap:
                m = min(chunk, n_bootstrap - done)
                idx = torch.randint(0, n, (m, n), generator=gen)
                means.append(d[idx].mean(dim=1))
                done += m
            boot = torch.cat(means).sort().values
            lo = boot[int(alpha / 2 * (boot.numel() - 1))]
            hi = boot[int((1 - alpha / 2) * (boot.numel() - 1))]
            out["ci"] = (float(lo), float(hi))
        return out

    def naive_ate(self, outcome: str | None = None) -> float:
        """Difference in group means, ignoring the pairing.

        Reported alongside :meth:`ate` as a design check: with a balanced
        paired frame the two agree, and a gap between them means rows were
        dropped asymmetrically.
        """
        y = self.column(outcome or self.outcome)
        t = self.column(self.treatment)
        return _mean(
            yi for yi, ti in zip(y, t, strict=True) if ti
        ) - _mean(yi for yi, ti in zip(y, t, strict=True) if not ti)

    def ate_by(
        self, modifier: str, outcome: str | None = None
    ) -> dict[Any, float]:
        """Stratified ATE — a dependency-free CATE.

        Enough to answer "does the leak concentrate near the cut?" without
        installing EconML. The DML estimator in :mod:`scaf.estimate.pywhy`
        gives the same quantity with confidence intervals and a continuous fit.
        """
        y = self.column(outcome or self.outcome)
        t = self.column(self.treatment)
        units = self.column("unit_id")
        strata = self.column(modifier)
        buckets: dict[Any, dict[str, dict[Any, float]]] = {}
        for yi, ti, ui, si in zip(y, t, units, strata, strict=True):
            b = buckets.setdefault(si, {"t": {}, "c": {}})
            b["t" if ti else "c"][ui] = yi
        out = {}
        for si, b in buckets.items():
            shared = b["t"].keys() & b["c"].keys()
            out[si] = _mean(b["t"][u] - b["c"][u] for u in shared)
        return dict(sorted(out.items(), key=_sort_key))

    def strata_counts(self, modifier: str) -> dict[Any, int]:
        """Pairs contributing to each stratum of :meth:`ate_by`.

        Worth reading next to the profile: a stratum backed by two pairs is
        not evidence of anything, and position sampling makes thin strata easy
        to produce without noticing.
        """
        t = self.column(self.treatment)
        units = self.column("unit_id")
        strata = self.column(modifier)
        seen: dict[Any, set] = {}
        for ti, ui, si in zip(t, units, strata, strict=True):
            if ti:
                seen.setdefault(si, set()).add(ui)
        return dict(
            sorted(((k, len(v)) for k, v in seen.items()), key=_sort_key)
        )

    # ------------------------------------------------------------------
    def add_within_unit(self, source: str, name: str | None = None) -> str:
        """Add a within-pair centred copy of ``source``, and return its name.

        Subtracting each pair's own mean is the fixed-effects trick, and here
        it is not a refinement but a correctness fix for *inference*. Raw NLL
        varies by many nats across positions — an early position with little
        context is simply harder than a late one — and that between-position
        spread has nothing to do with the treatment. An estimator that treats
        rows as independent charges all of it to residual noise, which inflates
        the standard error until a real leak reads as insignificant. For a
        library whose purpose is to prevent false all-clears, that is the
        dangerous direction to be wrong in.

        Centring removes it. The two rows of a pair become :math:`\\pm\\delta/2`
        around their own mean, so the point estimate is unchanged — the paired
        difference is still :math:`\\delta` — while the residual variance drops
        to the genuine heterogeneity of the effect.
        """
        name = name or f"{source}_within"
        y = self.column(source)
        units = self.column("unit_id")
        totals: dict[Any, list[float]] = {}
        for yi, ui in zip(y, units, strict=True):
            totals.setdefault(ui, []).append(yi)
        means = {u: sum(v) / len(v) for u, v in totals.items()}
        self.columns[name] = [
            yi - means[ui] for yi, ui in zip(y, units, strict=True)
        ]
        return name

    # ------------------------------------------------------------------
    # Graph
    # ------------------------------------------------------------------
    def to_dot(self, outcome: str | None = None) -> str:
        """The causal graph in DOT, accepted directly by DoWhy.

        Note what is *absent*: no edge points into the treatment. Writing one
        would assert a confounder we know does not exist, since the arm was
        assigned by the auditor rather than observed.

        Args:
            outcome: Name the graph's outcome node. Defaults to the frame's,
                but must be overridden when estimating on a derived column
                such as the within-pair centred outcome — DoWhy resolves the
                outcome by name against this graph and fails if it is absent.
        """
        outcome = outcome or self.outcome
        nodes = [self.treatment, outcome, *self.effect_modifiers]
        lines = ["digraph {"]
        lines += [f"  {n};" for n in nodes]
        lines.append(f"  {self.treatment} -> {outcome};")
        lines += [f"  {m} -> {outcome};" for m in self.effect_modifiers]
        lines += [f"  {c} -> {outcome};" for c in self.common_causes]
        lines += [f"  {c} -> {self.treatment};" for c in self.common_causes]
        lines.append("}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------
    def to_pandas(self):
        """Convert to a ``pandas.DataFrame``.

        Raises:
            ImportError: with install instructions if pandas is absent.
        """
        try:
            import pandas as pd
        except ImportError as exc:  # pragma: no cover - env dependent
            raise ImportError(
                "to_pandas() needs pandas: pip install 'semsimula-scaf[pywhy]'"
            ) from exc
        return pd.DataFrame(self.columns)

    def to_csv(self, path) -> None:
        """Write the frame as CSV, for tooling outside the PyWhy stack."""
        names = list(self.columns)
        with open(path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(names)
            w.writerows(zip(*(self.columns[n] for n in names), strict=True))

    # ------------------------------------------------------------------
    def summary(self, test: bool = True) -> str:
        ate = self.ate()
        lines = [
            f"LeakFrame — {len(self)} rows, {self.n_units} paired units",
            f"  treatment={self.treatment}  outcome={self.outcome} (nats)",
            f"  ATE (paired)    {ate:+.6g} nats",
            f"  ATE (naive)     {self.naive_ate():+.6g} nats",
        ]
        if test:
            r = self.ate_test(n_permutations=2000, n_bootstrap=1000)
            if r["ci"]:
                lines.append(
                    f"  95% CI          [{r['ci'][0]:+.6g}, {r['ci'][1]:+.6g}]"
                )
            lines.append(
                f"  sign-flip p     {r['p_value']:.4g}"
                f"   ({r['n_units']} pairs, exact)"
            )
        by_target = self.ate_by("target_perturbed")
        if len(by_target) > 1:
            lines.append("  by target_perturbed:")
            for k, v in by_target.items():
                label = (
                    "next token resampled (sharp)" if k
                    else "target unchanged (diffuse)"
                )
                lines.append(f"    {str(k):5s} {v:+.6g} nats   {label}")
        if f"{self.outcome}_within" in self.columns:
            lines.append(
                f"  outcome {self.outcome}_within carries the same ATE with "
                "the between-position variance removed"
            )
        return "\n".join(lines)

    def __str__(self) -> str:  # pragma: no cover - debugging aid
        return self.summary()


# ----------------------------------------------------------------------
def build_leak_frame(
    model,
    tokens=None,
    *,
    corpus: Corpus | None = None,
    adapter=None,
    device: str | torch.device = "cpu",
    dtype=None,
    seq_len: int = 128,
    n_seqs: int = 8,
    splits: tuple[float, ...] = (0.25, 0.5, 0.75),
    n_pairs: int = 2,
    max_positions: int = 16,
    first_frac: float = 0.1,
    micro_batch: int = 4,
    seed: int = 0,
    vocab_size: int | None = None,
    include_hidden_states: bool = False,
    include_basin_membership: bool = False,
) -> LeakFrame:
    """Run the future-perturbation intervention and record it row by row.

    The counterfactual futures come from
    :func:`~scaf.probes.future_perturbation.make_perturbation_pairs`, the same
    helper the probe uses, so a frame built with matching arguments describes
    exactly the intervention the probe measured. That correspondence is what
    lets the estimated ATE be checked against the probe's reported effect.

    Args:
        model: Model under test, or an existing
            :class:`~scaf.core.intervenable.InterventableModel`.
        tokens: Real token ids. Strongly preferred: a leak's magnitude depends
            on the counterfactual future being in-distribution.
        corpus: Explicit corpus, overriding ``tokens``.
        adapter: Explicit adapter; auto-detected when omitted.
        device: Device to run on.
        dtype: Cast the model before probing; ``None`` leaves it alone.
        seq_len: Probe sequence length.
        n_seqs: Sequences per split.
        splits: Sequence fractions at which to cut the future.
        n_pairs: Counterfactual futures per split.
        max_positions: Target positions scored per split. The main cost and
            row-count knob.
        first_frac: Earliest scored position, as a fraction of the sequence.
            Very early positions have almost no context, so their NLL is
            dominated by the unigram prior rather than by any leak.
        micro_batch: Forward chunk size; ``0`` disables chunking.
        seed: Corpus seed.
        vocab_size: Needed only for a synthetic corpus when the adapter cannot
            read a vocabulary size from the model config.
        include_hidden_states: When ``True`` and the adapter exposes
            ``has_hidden_states``, the frame gains ``hidden_cos_dev`` and
            ``layer`` columns — one row per (layer, position) rather than per
            position alone. ``layer`` is added to ``effect_modifiers`` so
            CATE-by-layer analysis works out of the box. When ``False``
            (default) or the adapter lacks trajectory support, the frame is
            unchanged.
        include_basin_membership: When ``True`` and the adapter exposes
            ``has_vtheta_wells``, the frame gains ``basin_changed``,
            ``well_id_factual``, and ``well_id_counterfactual`` columns.
            Implies ``include_hidden_states=True`` (basin assignment
            requires per-layer hidden states). ``basin_changed`` is added
            to ``effect_modifiers`` for CATE stratification.

    Returns:
        A :class:`LeakFrame`.
    """
    owns = not isinstance(model, InterventableModel)
    im = (
        InterventableModel(model, adapter=adapter, device=device, dtype=dtype)
        if owns
        else model
    )
    try:
        cfg = im.config()
        if corpus is None:
            if tokens is not None:
                corpus = TokenCorpus(
                    tokens, seq_len=seq_len, seed=seed, vocab_size=vocab_size
                )
            else:
                v = vocab_size or cfg.get("vocab_size")
                if not v:
                    raise ValueError(
                        "no tokens given and vocab_size could not be inferred "
                        "from the model config; pass tokens= or vocab_size="
                    )
                corpus = SyntheticCorpus(v, seq_len=seq_len, seed=seed)

        x, pairs = make_perturbation_pairs(
            corpus, n_seqs, splits, n_pairs, im.device
        )
        T = int(x.shape[1])

        if include_basin_membership:
            include_hidden_states = True
        use_hidden = (
            include_hidden_states and im.caps.has_hidden_states
        )
        use_basins = (
            include_basin_membership
            and use_hidden
            and im.caps.has_vtheta_wells
        )

        col_names = [
            "unit_id", "future_perturbed", "nll", "logit_l1",
            "position", "position_frac", "distance_to_cut",
            "split_frac", "target_perturbed", "seq_id", "pair_id",
        ]
        if use_hidden:
            col_names += ["hidden_cos_dev", "layer"]
        if use_basins:
            col_names += [
                "basin_changed", "well_id_factual", "well_id_counterfactual",
            ]
        cols: dict[str, list] = {k: [] for k in col_names}
        n_skipped_splits = 0

        with im.deterministic():
            base = _chunked_logits(im, x, micro_batch)
            x_cpu = x.to(base.device)
            base_traj = None
            if use_hidden:
                _, base_traj = _chunked_trajectory(im, x, micro_batch)

            seen: dict[tuple[float, int], int] = {}
            for frac, t_p, x_cf in pairs:
                pair_id = seen[(frac, t_p)] = seen.get((frac, t_p), -1) + 1
                cf = _chunked_logits(im, x_cf, micro_batch)
                cf_traj = None
                if use_hidden:
                    _, cf_traj = _chunked_trajectory(im, x_cf, micro_batch)

                lo = max(1, int(round(first_frac * (T - 1))))
                hi = t_p
                if hi < lo:
                    n_skipped_splits += 1
                    continue
                n = min(max_positions, hi - lo + 1)
                positions = sorted(
                    set(
                        torch.linspace(lo, hi, steps=n)
                        .round().long().tolist()
                    )
                )

                # Pre-compute per-layer cosine deviation for all prefix
                # positions in one pass (only the needed positions are
                # indexed below).
                cos_dev_by_layer: list[torch.Tensor] | None = None
                if use_hidden and base_traj is not None and cf_traj is not None:
                    cos_dev_by_layer = [
                        _cosine_deviation(
                            base_traj[ell][:, : t_p + 1],
                            cf_traj[ell][:, : t_p + 1],
                        )
                        for ell in range(len(base_traj))
                    ]

                # Pre-compute per-layer dominant well assignments.
                wells_f_by_layer: list[torch.Tensor] | None = None
                wells_c_by_layer: list[torch.Tensor] | None = None
                if use_basins and base_traj is not None and cf_traj is not None:
                    wells_f_by_layer = []
                    wells_c_by_layer = []
                    for ell in range(len(base_traj)):
                        wp = im.adapter.well_parameters(im.model, ell, x)
                        if wp is not None:
                            wf = assign_dominant_wells(
                                base_traj[ell][:, : t_p + 1],
                                wp["mu"], wp["precision_diag"],
                                wp["precision_lr"], wp["weights"],
                            )
                            wc = assign_dominant_wells(
                                cf_traj[ell][:, : t_p + 1],
                                wp["mu"], wp["precision_diag"],
                                wp["precision_lr"], wp["weights"],
                            )
                            wells_f_by_layer.append(wf)
                            wells_c_by_layer.append(wc)
                        else:
                            B_sz = base_traj[ell][:, : t_p + 1].shape[:2]
                            wells_f_by_layer.append(torch.zeros(B_sz, dtype=torch.long))
                            wells_c_by_layer.append(torch.zeros(B_sz, dtype=torch.long))

                layers = (
                    list(range(len(base_traj)))
                    if use_hidden and base_traj is not None
                    else [None]
                )

                for t in positions:
                    target = x_cpu[:, t + 1]
                    nll_f = F.cross_entropy(
                        base[:, t].float(), target, reduction="none"
                    )
                    nll_c = F.cross_entropy(
                        cf[:, t].float(), target, reduction="none"
                    )
                    l1 = (cf[:, t] - base[:, t]).abs().sum(-1).float()

                    for ell in layers:
                        for b in range(x.shape[0]):
                            layer_tag = (
                                f"_L{ell}" if ell is not None else ""
                            )
                            unit = (
                                f"s{b}_c{t_p}_p{pair_id}_t{t}{layer_tag}"
                            )
                            for arm, nll, dev in (
                                (0, float(nll_f[b]), 0.0),
                                (1, float(nll_c[b]), float(l1[b])),
                            ):
                                cols["unit_id"].append(unit)
                                cols["future_perturbed"].append(arm)
                                cols["nll"].append(nll)
                                cols["logit_l1"].append(dev)
                                cols["position"].append(int(t))
                                cols["position_frac"].append(t / (T - 1))
                                cols["distance_to_cut"].append(
                                    int(t_p - t)
                                )
                                cols["split_frac"].append(float(frac))
                                cols["target_perturbed"].append(
                                    int(t + 1 > t_p)
                                )
                                cols["seq_id"].append(int(b))
                                cols["pair_id"].append(int(pair_id))
                                if use_hidden:
                                    cos_val = (
                                        float(cos_dev_by_layer[ell][b, t])
                                        if arm == 1
                                        and cos_dev_by_layer is not None
                                        and t < cos_dev_by_layer[ell].shape[1]
                                        else 0.0
                                    )
                                    cols["hidden_cos_dev"].append(cos_val)
                                    cols["layer"].append(
                                        ell if ell is not None else -1
                                    )
                                if use_basins:
                                    if (
                                        wells_f_by_layer is not None
                                        and wells_c_by_layer is not None
                                        and ell is not None
                                        and t < wells_f_by_layer[ell].shape[-1]
                                    ):
                                        wf_val = int(wells_f_by_layer[ell][b, t])
                                        wc_val = int(wells_c_by_layer[ell][b, t])
                                    else:
                                        wf_val = -1
                                        wc_val = -1
                                    cols["basin_changed"].append(
                                        int(wf_val != wc_val)
                                    )
                                    cols["well_id_factual"].append(wf_val)
                                    cols["well_id_counterfactual"].append(wc_val)

        if not cols["unit_id"]:
            raise ValueError(
                f"no scoreable positions at seq_len={T} with splits={splits} "
                f"and first_frac={first_frac}: every cut falls inside the "
                "warm-up region. Use a longer seq_len or later splits."
            )

        modifiers = (
            "position", "split_frac", "target_perturbed",
            "distance_to_cut",
        )
        if use_hidden:
            modifiers = modifiers + ("layer",)
        if use_basins:
            modifiers = modifiers + ("basin_changed",)

        frame = LeakFrame(
            columns=cols,
            effect_modifiers=modifiers,
            metadata={
                "model": type(im.model).__name__,
                "adapter": im.adapter.name,
                "dtype": str(im.dtype),
                "device": str(im.device),
                "seq_len": T,
                "n_seqs": n_seqs,
                "splits": list(splits),
                "n_pairs": n_pairs,
                "max_positions": max_positions,
                "skipped_splits": n_skipped_splits,
                "include_hidden_states": use_hidden,
                "include_basin_membership": use_basins,
                "corpus": type(corpus).__name__,
                "config": cfg,
            },
        )
        frame.add_within_unit("nll")
        frame.add_within_unit("logit_l1")
        return frame
    finally:
        if owns:
            im.close()
