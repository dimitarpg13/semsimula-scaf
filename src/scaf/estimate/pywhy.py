"""DoWhy / EconML bridge — formal estimands, refutation, and CATE.

**What this buys, and what it does not.** DoWhy's headline capability is
*identification*: deciding, from a graph, whether an effect is estimable from
observational data despite confounding. SCAF gets no value from that, because
there is no confounding to defeat — the auditor assigns the arm, so the naive
paired difference in :meth:`~scaf.estimate.frames.LeakFrame.ate` is already the
causal effect. Pretending otherwise would be theatre.

Three things are genuinely worth importing a heavy stack for:

1. **Refutation.** ``placebo_treatment_refuter`` and friends are a
   standardised, citable robustness suite. The hand-written probes
   approximated them; here they are the reference implementation, run the same
   way a reviewer would run them.
2. **Heterogeneity.** EconML fits :math:`\\tau(x)` continuously with confidence
   intervals, answering "*which* positions leak" rather than "does it leak".
   For the register leak the expected shape is sharply increasing as the target
   approaches the cut, and a flat profile would mean the diagnosis is wrong.
3. **Cross-validation of SCAF itself.** An independent estimator reproducing
   the paired difference is evidence the frame is built correctly. This module
   therefore always reports both and flags disagreement, treating a mismatch as
   a bug in the setup rather than as a subtler finding.

**On the placebo refuter.** It permutes the treatment column, which destroys
the factual/counterfactual pairing that carries the entire signal. That is
precisely why it is a valid placebo: after permutation the rows are the same
numbers with the arm label scrambled, and any surviving effect is an artefact
of the estimator rather than of the intervention.

Everything here is optional. ``pip install 'semsimula-scaf[pywhy]'``; the core
probe battery and :class:`~scaf.monitor.LeakMonitor` never touch it.
"""

from __future__ import annotations

import contextlib
import logging
import warnings
from dataclasses import dataclass, field
from typing import Any

from .frames import LeakFrame

__all__ = ["EstimationReport", "RefutationResult", "estimate_leak"]

_INSTALL_HINT = (
    "the DoWhy/EconML bridge needs optional dependencies: "
    "pip install 'semsimula-scaf[pywhy]'"
)

#: Refuter name to how its result should be read. ``"null"`` means the effect
#: must collapse to zero; ``"stable"`` means it must stay close to the original.
_REFUTER_SEMANTICS = {
    "placebo_treatment_refuter": "null",
    "random_common_cause": "stable",
    "data_subset_refuter": "stable",
    "bootstrap_refuter": "stable",
}


def _require(module: str):
    try:
        return __import__(module)
    except ImportError as exc:  # pragma: no cover - env dependent
        raise ImportError(f"{_INSTALL_HINT} (missing {module})") from exc


@contextlib.contextmanager
def _quiet(enabled: bool):
    """Silence DoWhy/EconML's very chatty INFO logging and sklearn warnings.

    Suppressed by default because a single estimate emits dozens of lines that
    bury the numbers. Pass ``verbose=True`` when debugging an estimator.
    """
    if not enabled:
        yield
        return
    names = ["dowhy", "econml", "sklearn", "py.warnings"]
    saved = {n: logging.getLogger(n).level for n in names}
    for n in names:
        logging.getLogger(n).setLevel(logging.ERROR)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            yield
    finally:
        for n, lvl in saved.items():
            logging.getLogger(n).setLevel(lvl)


@dataclass
class RefutationResult:
    """One robustness check, with its pass/fail reading already applied."""

    name: str
    original_effect: float
    new_effect: float
    p_value: float | None = None
    semantics: str = "stable"
    passed: bool | None = None
    error: str | None = None

    def __str__(self) -> str:
        if self.error:
            return f"[SKIP] {self.name}: {self.error}"
        status = {True: "PASS", False: "FAIL", None: "SKIP"}[self.passed]
        expect = (
            "expected ~0" if self.semantics == "null"
            else f"expected ~{self.original_effect:.4g}"
        )
        return (
            f"[{status}] {self.name}: new effect {self.new_effect:+.4g} "
            f"({expect})"
        )


@dataclass
class EstimationReport:
    """Result of the identify / estimate / refute / heterogeneity workflow."""

    ate: float = 0.0
    stderr: float | None = None
    ci: tuple[float, float] | None = None
    p_value: float | None = None
    #: The frame's exact mean paired difference. The estimator should match it.
    reference_ate: float = 0.0
    #: Sign-flip permutation p-value from
    #: :meth:`~scaf.estimate.frames.LeakFrame.ate_test`. This is the primary
    #: inference: it is exact under the paired design, where DoWhy's pooled
    #: permutation is not.
    exact_p_value: float | None = None
    exact_ci: tuple[float, float] | None = None
    estimand: str = ""
    cate: dict[str, Any] = field(default_factory=dict)
    refutations: list[RefutationResult] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    #: Relative tolerance for agreeing with ``reference_ate``.
    rtol: float = 0.05
    atol: float = 1e-6

    @property
    def agrees_with_reference(self) -> bool:
        gap = abs(self.ate - self.reference_ate)
        return gap <= self.atol + self.rtol * abs(self.reference_ate)

    @property
    def refutations_ok(self) -> bool:
        ran = [r for r in self.refutations if r.passed is not None]
        return bool(ran) and all(r.passed for r in ran)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ate": self.ate,
            "stderr": self.stderr,
            "ci": list(self.ci) if self.ci else None,
            "p_value": self.p_value,
            "exact_p_value": self.exact_p_value,
            "exact_ci": list(self.exact_ci) if self.exact_ci else None,
            "reference_ate": self.reference_ate,
            "agrees_with_reference": self.agrees_with_reference,
            "refutations": [
                {
                    "name": r.name,
                    "original_effect": r.original_effect,
                    "new_effect": r.new_effect,
                    "p_value": r.p_value,
                    "passed": r.passed,
                    "error": r.error,
                }
                for r in self.refutations
            ],
            "cate": self.cate,
            "errors": self.errors,
            "metadata": self.metadata,
        }

    def summary(self) -> str:
        lines = [
            "SCAF estimand — do(future) on next-token NLL",
            f"  ATE            {self.ate:+.6g} nats",
        ]
        if self.exact_ci:
            lines.append(
                f"  95% CI         [{self.exact_ci[0]:+.6g}, "
                f"{self.exact_ci[1]:+.6g}]   (paired bootstrap)"
            )
        if self.exact_p_value is not None:
            lines.append(
                f"  p-value        {self.exact_p_value:.4g}"
                "   (sign-flip, exact under pairing)"
            )
        if self.ci:
            lines.append(
                f"  DoWhy 95% CI   [{self.ci[0]:+.6g}, {self.ci[1]:+.6g}]"
            )
        if self.p_value is not None:
            lines.append(
                f"  DoWhy p-value  {self.p_value:.3g}   (pooled permutation; "
                "ignores pairing, so treat as a lower bound on significance)"
            )
        lines.append(
            f"  paired ref.    {self.reference_ate:+.6g} nats"
            f"   ({'agrees' if self.agrees_with_reference else 'DISAGREES'})"
        )
        if not self.agrees_with_reference:
            lines.append(
                "  warning: the estimator disagrees with the exact paired "
                "difference. Treatment here is randomised, so the two must "
                "match — suspect the estimator configuration, not the data."
            )
        if self.refutations:
            lines += ["", "  Refutations:"]
            lines += [f"    {r}" for r in self.refutations]
        lines += self._cate_lines()
        for e in self.errors:
            lines.append(f"  note: {e}")
        return "\n".join(lines)

    def _cate_lines(self) -> list[str]:
        if not self.cate:
            return []
        out = ["", "  Heterogeneity (exact stratified profile):"]
        for axis, entry in self.cate.items():
            exact = entry.get("exact") or []
            if not exact:
                continue
            fit = entry.get("fit") or {}
            fitted = dict(fit.get("points", []))
            head = f"    tau by {axis}:"
            if fitted:
                head += f"   [{fit.get('model', 'fit')} in brackets]"
            out.append(head)
            for xv, tv, n in exact:
                row = f"      {axis}={xv:<7g} {tv:+11.4g} nats  ({n:>3d} pairs)"
                if xv in fitted:
                    row += f"   [{fitted[xv]:+.4g}]"
                out.append(row)
            if fitted:
                gap = max(
                    abs(tv - fitted[xv]) for xv, tv, _ in exact if xv in fitted
                )
                out.append(
                    f"      fit deviates from exact by at most {gap:.4g} nats"
                )
            peak = max(exact, key=lambda r: abs(r[1]))
            out.append(
                f"      peak at {axis}={peak[0]:g} ({peak[1]:+.4g} nats)"
            )
        return out

    def __str__(self) -> str:  # pragma: no cover - debugging aid
        return self.summary()


# ----------------------------------------------------------------------
#: EconML estimators exposed through the bridge.
#:
#: ``forest`` is the default because leak profiles are typically *spiky* rather
#: than smooth: a shared-register leak can be worth a hundred nats at the
#: position adjacent to the cut and exactly zero two positions away. A linear
#: CATE cannot represent that. Fitted to the peeking reference model it reports
#: a third of the true effect at the spike and a spurious *negative* effect far
#: from it, which would read as the future suppressing the past — a conclusion
#: the data does not contain. The causal forest recovers the spike to within
#: 1%, and does it faster.
_CATE_MODELS = {
    "forest": (
        "backdoor.econml.dml.CausalForestDML",
        {"discrete_treatment": True, "cv": 2, "n_estimators": 200},
    ),
    "linear": (
        "backdoor.econml.dml.LinearDML",
        {"discrete_treatment": True, "cv": 2},
    ),
}


def _fit_cate(model, estimand, df, axis, kind, random_seed):
    """Fit an EconML CATE along one effect modifier."""
    method, init = _CATE_MODELS[kind]
    est = model.estimate_effect(
        estimand,
        method_name=method,
        control_value=0,
        treatment_value=1,
        target_units="ate",
        # The treatment is binary and its propensity is exactly 0.5 by
        # construction, so declaring it discrete keeps EconML from fitting a
        # regression where a constant would do.
        method_params={
            "init_params": {**init, "random_state": random_seed},
            "fit_params": {},
        },
        effect_modifiers=[axis],
    )
    xs = sorted(set(df[axis].tolist()))
    # Evaluate on the observed grid rather than a synthetic linspace, so every
    # reported tau corresponds to a position actually measured.
    taus = (
        est.estimator.estimator
        .effect(_as_2d([[float(v)] for v in xs]))
        .reshape(-1).tolist()
    )
    return {
        "model": method.rsplit(".", 1)[-1],
        "points": list(zip([float(v) for v in xs], taus, strict=True)),
        "ate": float(est.value),
    }


def _as_2d(rows):
    """Rows of floats to the 2-D array EconML expects, without importing numpy.

    NumPy is present transitively whenever EconML is, but SCAF avoids importing
    it directly so the core stays free of the torch/NumPy version bridge. Here
    we are already inside the optional extra, so a local import is honest.
    """
    import numpy as np

    return np.asarray(rows, dtype=float)


def estimate_leak(
    frame: LeakFrame,
    *,
    outcome: str | None = None,
    cate_axes: tuple[str, ...] = ("distance_to_cut",),
    cate_model: str | None = "forest",
    refuters: tuple[str, ...] = (
        "placebo_treatment_refuter",
        "random_common_cause",
    ),
    num_simulations: int = 20,
    refute_tol: float = 0.1,
    dowhy_inference: bool = False,
    quiet: bool = True,
    random_seed: int = 0,
) -> EstimationReport:
    """Run DoWhy's four-verb workflow over a :class:`LeakFrame`.

    Args:
        frame: Frame from :func:`~scaf.estimate.frames.build_leak_frame`.
        outcome: Outcome column; defaults to the frame's (``nll``, in nats).
        cate_axes: Axes to profile heterogeneous effects along. Any frame
            column works. ``distance_to_cut`` is the diagnostic axis for a
            register leak: a shared-state leak spikes right at the cut and
            decays away from it, a next-token peek is a spike and nothing
            else, and a global-pool leak is roughly flat. An exact stratified
            profile is always reported; the EconML fit is a smoother over it.
        cate_model: ``"forest"`` (default), ``"linear"``, or ``None`` to skip
            the EconML fit and report only the exact stratified profile. See
            :data:`_CATE_MODELS` for why the forest is the default.
        refuters: DoWhy refuter names to run.
        num_simulations: Simulations per refuter. The default is small enough
            to keep an audit interactive; raise it for a published number.
        refute_tol: Fractional tolerance for reading a refutation. Compared
            against the magnitude of the original effect, so it scales.
        dowhy_inference: Also compute DoWhy's own standard error, confidence
            interval, and significance test. Off by default for two reasons.
            It dominates the runtime — each is a bootstrap that refits the
            estimator a hundred times, turning a four-second call into a
            ninety-second one. And its significance test permutes treatment
            across the pooled frame, which discards the pairing that carries
            the signal and can return a non-significant p-value for an effect
            its own interval puts nowhere near zero. The exact paired
            statistics from
            :meth:`~scaf.estimate.frames.LeakFrame.ate_test` are always
            reported and are the ones to quote; enable this only to show the
            two approaches side by side.
        quiet: Suppress DoWhy/EconML INFO logging.
        random_seed: Seed for the refuters' resampling.

    Returns:
        An :class:`EstimationReport`. Individual stages degrade to an entry in
        ``errors`` rather than raising, so a failed CATE fit never costs you
        the ATE and the refutations.

    Raises:
        ImportError: if the ``pywhy`` extra is not installed.
    """
    _require("dowhy")
    _require("pandas")
    from dowhy import CausalModel

    # The within-pair centred outcome is preferred when available: it carries
    # the identical ATE but strips the between-position variance that would
    # otherwise inflate every standard error. See
    # :meth:`~scaf.estimate.frames.LeakFrame.add_within_unit`.
    default_outcome = frame.outcome
    within = f"{frame.outcome}_within"
    if within in frame.columns:
        default_outcome = within
    outcome = outcome or default_outcome
    exact = frame.ate_test(outcome, seed=random_seed)
    report = EstimationReport(
        reference_ate=exact["ate"],
        exact_p_value=exact["p_value"],
        exact_ci=exact["ci"],
        metadata={
            "n_rows": len(frame),
            "n_units": frame.n_units,
            "outcome": outcome,
            "exact_stderr": exact["stderr"],
            **frame.metadata,
        },
    )

    df = frame.to_pandas()
    keep = {
        frame.treatment, outcome,
        *frame.effect_modifiers, *frame.common_causes, *cate_axes,
    }
    # DoWhy treats every dataframe column as a potential variable and warns
    # about the ones missing from the graph. Bookkeeping columns like unit_id
    # are not causal variables, so they are dropped rather than declared.
    df = df[[c for c in df.columns if c in keep]]

    with _quiet(quiet):
        model = CausalModel(
            data=df,
            treatment=frame.treatment,
            outcome=outcome,
            graph=frame.to_dot(outcome),
            common_causes=list(frame.common_causes) or None,
            effect_modifiers=list(frame.effect_modifiers),
        )
        estimand = model.identify_effect(proceed_when_unidentifiable=True)
        report.estimand = str(estimand)

        estimate = model.estimate_effect(
            estimand,
            method_name="backdoor.linear_regression",
            control_value=0,
            treatment_value=1,
            test_significance=dowhy_inference,
        )
        report.ate = float(estimate.value)
        if dowhy_inference:
            # Each of these triggers its own bootstrap inside DoWhy, so they
            # are requested together or not at all.
            report.stderr = _maybe(estimate, "get_standard_error")
            report.ci = _confidence_interval(estimate)
            report.p_value = _significance(estimate)

        for axis in cate_axes:
            if axis not in frame.columns:
                report.errors.append(
                    f"cate axis {axis!r} is not a column of the frame"
                )
                continue
            # The stratified profile is exact, model-free, and free of charge,
            # so it is computed whether or not the EconML fit succeeds. It is
            # also the thing to trust: the fit is a smoother over it, useful
            # for predicting unmeasured positions, not a better measurement of
            # the measured ones.
            exact_profile = frame.ate_by(axis, outcome)
            counts = frame.strata_counts(axis)
            entry: dict[str, Any] = {
                "exact": [
                    (float(k), v, counts.get(k, 0))
                    for k, v in exact_profile.items()
                ],
            }
            if cate_model:
                try:
                    entry["fit"] = _fit_cate(
                        model, estimand, df, axis, cate_model, random_seed
                    )
                except Exception as exc:  # noqa: BLE001 - report, never abort
                    report.errors.append(f"CATE on {axis!r} failed: {exc}")
            report.cate[axis] = entry

        for name in refuters:
            report.refutations.append(
                _run_refuter(
                    model, estimand, estimate, name,
                    num_simulations, refute_tol, random_seed,
                )
            )

    return report


def _run_refuter(
    model, estimand, estimate, name, num_simulations, tol, seed
) -> RefutationResult:
    semantics = _REFUTER_SEMANTICS.get(name, "stable")
    original = float(estimate.value)
    try:
        kwargs: dict[str, Any] = {
            "method_name": name,
            "random_seed": seed,
        }
        if name != "random_common_cause":
            kwargs["num_simulations"] = num_simulations
        if name == "placebo_treatment_refuter":
            kwargs["placebo_type"] = "permute"
        res = model.refute_estimate(estimand, estimate, **kwargs)
        new = float(res.new_effect)
        p = getattr(res, "refutation_result", None)
        p_value = (
            float(p["p_value"]) if isinstance(p, dict) and "p_value" in p
            else None
        )
    except Exception as exc:  # noqa: BLE001 - a refuter must not abort a run
        return RefutationResult(
            name=name, original_effect=original, new_effect=float("nan"),
            semantics=semantics, error=str(exc),
        )

    band = tol * max(abs(original), 1e-9)
    passed = (
        abs(new) <= band if semantics == "null"
        else abs(new - original) <= band
    )
    return RefutationResult(
        name=name, original_effect=original, new_effect=new,
        p_value=p_value, semantics=semantics, passed=bool(passed),
    )


def _maybe(estimate, method):
    """Call an optional DoWhy accessor, tolerating version differences."""
    fn = getattr(estimate, method, None)
    if fn is None:
        return None
    try:
        v = fn()
    except Exception:  # noqa: BLE001 - accessor availability varies by version
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return float(v[0]) if hasattr(v, "__len__") and len(v) else None


def _significance(estimate):
    sig = getattr(estimate, "test_stat_significance", None)
    if callable(sig):
        try:
            return float(sig()["p_value"][0])
        except Exception:  # noqa: BLE001 - shape varies across versions
            pass
    raw = getattr(estimate, "significance_test", None)
    if isinstance(raw, dict) and "p_value" in raw:
        try:
            p = raw["p_value"]
            return float(p[0] if hasattr(p, "__len__") else p)
        except (TypeError, ValueError, IndexError):
            return None
    return None


def _confidence_interval(estimate):
    fn = getattr(estimate, "get_confidence_intervals", None)
    if fn is None:
        return None
    try:
        ci = fn()
    except Exception:  # noqa: BLE001 - not all estimators support intervals
        return None
    try:
        flat = [float(v) for v in _flatten(ci)]
    except (TypeError, ValueError):
        return None
    return (flat[0], flat[1]) if len(flat) >= 2 else None


def _flatten(x):
    if hasattr(x, "tolist"):
        x = x.tolist()
    if isinstance(x, (list, tuple)):
        for i in x:
            yield from _flatten(i)
    else:
        yield x
