"""``LeakMonitor`` — continuous leak surveillance inside a training loop.

A one-off audit certifies a checkpoint. It does not certify a *run*, and the
distinction is not academic. A leak channel is usually gated by a learnable
scale initialised near zero, so the architecture leaks from step one while the
measured effect is nil. The valve opens gradually as the optimiser discovers
that the channel pays. By the time the final checkpoint is audited, the leak is
fully open and every intermediate number in the training log is already
contaminated. Nothing in the loss curve marks the moment it happened.

This class is the fix: a small probe battery, run on a fixed set of sequences
at a fixed step interval, so the leak's *trajectory* is recorded alongside the
loss. When a run turns out to be compromised, the log says at which step the
valve opened and what it was worth in nats at every point.

Three properties make it safe to leave switched on.

**It does not disturb the run.** Global RNG state is saved and restored around
every measurement, so a monitored run follows the exact same trajectory as an
unmonitored one. Without that, turning monitoring on would silently change
which batches, dropout masks, and routing samples the model sees, and two runs
that differ only in observability would diverge. Training mode and dtype are
restored too, and hooks are registered per measurement rather than left
installed, so the training forward pass is untouched between probes.

**It does not crash the run.** A monitor that kills a 1000-minute job over an
out-of-memory error in a diagnostic is worse than no monitor. Failures are
caught and recorded as an ``ERROR`` verdict; training continues. Aborting is
available, but only for an actual leak and only when asked for explicitly.

**It probes the same sequences every time.** The corpus is rewound before each
run, so a change in AILE between step 20k and step 40k is a change in the
model, not a change in the draw.
"""

from __future__ import annotations

import contextlib
import json
import time
from typing import Any

import torch

from .controls import DeterminismControl, PlaceboControl, PositiveControl
from .core.corpus import Corpus, SyntheticCorpus, TokenCorpus
from .core.intervenable import InterventableModel
from .probes.future_perturbation import FuturePerturbationProbe
from .probes.target_relocation import TargetRelocationProbe
from .report import CausalLeakError, LeakScorecard

__all__ = ["LeakMonitor"]


@contextlib.contextmanager
def preserved_rng():
    """Restore global RNG state on exit, on CPU and every CUDA device.

    The monitor draws probe sequences and perturbations, and the model's own
    forward may consume randomness through dropout or stochastic routing. Left
    unrestored, those draws would shift every subsequent training batch and
    mask, so enabling monitoring would change the run it is supposed to be
    observing. Corpora already use private generators; this covers the model.
    """
    cpu = torch.get_rng_state()
    cuda = (
        torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    )
    try:
        yield
    finally:
        torch.set_rng_state(cpu)
        if cuda is not None:
            torch.cuda.set_rng_state_all(cuda)


class LeakMonitor:
    """Periodic leak probing for a model that is still training.

    Usage mirrors the design note::

        monitor = scaf.LeakMonitor(
            model, tokens=val_tokens, interval=20_000,
            seq_len=512, micro_batch=2, jsonl_path=report_path,
        )
        for step in range(total_steps):
            ...
            monitor.maybe_run(step)

    Args:
        model: The model being trained. Held by reference and re-probed in
            place, so no reload is needed as weights change.
        tokens: Validation token ids. Strongly preferred over a synthetic
            corpus: the honest-perplexity stage is only meaningful on real
            text, and a leak's size depends on the counterfactual future being
            in-distribution.
        corpus: Explicit corpus, overriding ``tokens``.
        adapter: Explicit adapter; auto-detected when omitted.
        interval: Steps between runs. Measured from the last run rather than
            by divisibility, so a resumed run probes on schedule instead of
            waiting for the next multiple.
        run_at_start: Probe on the first call. Recommended — it records the
            architectural verdict before training can open any valve, which is
            the baseline every later measurement is read against.
        seq_len: Probe sequence length. Match the training block size; a leak
            carried by a bounded-range register can be invisible at a shorter
            context.
        n_seqs: Sequences per probe. Small on purpose.
        splits: Where to cut the future. One mid-sequence cut is enough for
            surveillance; the full battery in :func:`~scaf.api.audit` uses
            three.
        n_pairs: Counterfactual futures per split.
        micro_batch: Forward chunk size. Register tensors are ``(B, T, M, d)``
            under the prefix-causal fix, so at ``seq_len=512`` this wants to be
            ``2`` or ``1``.
        honest_ppl: Also run the target-relocation stage, which reports the
            leak tax in nats and the honest perplexity. This is the stage that
            exposed the original register leak after the perturbation test
            alone had understated it.
        honest_ppl_interval: Step interval for that second stage, when it
            should be rarer than the first. Defaults to ``interval``.
        n_targets: Target positions for the honest-perplexity stage. The
            dominant cost; each one is an extra forward pass.
        controls: Run the determinism, placebo, and positive controls. Leaving
            them on is what keeps a quiet probe from being read as a clean
            model. Disabling them makes every verdict ``INVALID``, by design.
        relocation_threshold: Tolerated honest-PPL gap in nats.
        dtype: Cast the model before probing. Almost always leave as ``None``:
            casting a real checkpoint to float64 mid-run is slow and can
            exhaust memory, and the determinism control measures the actual
            noise floor anyway.
        jsonl_path: Append each record here as one JSON line. Convenient for
            writing straight into the same report file the trainer uses.
        raise_on_leak: Abort the run by raising
            :class:`~scaf.report.CausalLeakError` when a leak is detected.
            Off by default: the usual preference is to finish the run with the
            contamination documented rather than lose it.
        seed: Corpus seed.
        vocab_size: Needed only for a synthetic corpus when the adapter cannot
            read a vocabulary size from the model config.
        device: Device for probing; defaults to wherever the model already is.
    """

    def __init__(
        self,
        model,
        tokens=None,
        *,
        corpus: Corpus | None = None,
        adapter=None,
        interval: int = 20_000,
        run_at_start: bool = True,
        seq_len: int = 128,
        n_seqs: int = 4,
        splits: tuple[float, ...] = (0.5,),
        n_pairs: int = 1,
        micro_batch: int = 2,
        honest_ppl: bool = True,
        honest_ppl_interval: int | None = None,
        n_targets: int = 16,
        controls: bool = True,
        relocation_threshold: float = 1e-3,
        dtype=None,
        jsonl_path=None,
        raise_on_leak: bool = False,
        seed: int = 0,
        vocab_size: int | None = None,
        device=None,
    ) -> None:
        if interval <= 0:
            raise ValueError(f"interval must be positive, got {interval}")

        self.model = model
        self.adapter = adapter
        self.interval = int(interval)
        self.run_at_start = run_at_start
        self.n_seqs = n_seqs
        self.splits = splits
        self.n_pairs = n_pairs
        self.micro_batch = micro_batch
        self.honest_ppl = honest_ppl
        self.honest_ppl_interval = int(honest_ppl_interval or interval)
        self.n_targets = n_targets
        self.controls = controls
        self.relocation_threshold = relocation_threshold
        self.dtype = dtype
        self.jsonl_path = jsonl_path
        self.raise_on_leak = raise_on_leak
        self.device = device if device is not None else _device_of(model)

        self.corpus = corpus or _build_corpus(
            model, tokens, seq_len, seed, vocab_size, adapter
        )

        self.history: list[dict[str, Any]] = []
        self.last_scorecard: LeakScorecard | None = None
        self._last_step: int | None = None
        self._last_honest_step: int | None = None

    # ------------------------------------------------------------------
    @property
    def first_leak_step(self) -> int | None:
        """Step at which a leak was first detected, if ever.

        The number to quote when explaining which parts of a training log are
        trustworthy: everything from here on was measured through the leak.
        """
        for r in self.history:
            if r.get("verdict") == "LEAK":
                return r["step"]
        return None

    @property
    def aile_trend(self) -> list[tuple[int, float]]:
        """``(step, aile)`` over the run — the valve-opening curve."""
        return [
            (r["step"], r["aile"])
            for r in self.history
            if r.get("aile") is not None
        ]

    # ------------------------------------------------------------------
    def due(self, step: int) -> bool:
        """Whether :meth:`run` would fire at ``step``."""
        if self._last_step is None:
            return bool(self.run_at_start)
        return (step - self._last_step) >= self.interval

    def maybe_run(self, step: int) -> dict[str, Any] | None:
        """Probe if due, else return ``None``.

        Safe to call every step; the schedule check is a comparison.
        """
        if self._last_step is None and not self.run_at_start:
            # Anchor the schedule so the first run lands one interval from
            # here, rather than never firing.
            self._last_step = int(step)
            return None
        if not self.due(step):
            return None
        return self.run(step)

    # ------------------------------------------------------------------
    def run(self, step: int) -> dict[str, Any]:
        """Probe now, regardless of schedule, and record the result."""
        step = int(step)
        t0 = time.time()
        want_honest = self.honest_ppl and (
            self._last_honest_step is None
            or (step - self._last_honest_step) >= self.honest_ppl_interval
        )

        try:
            record = self._measure(step, want_honest)
        except Exception as exc:  # noqa: BLE001 - never kill a training run
            record = {
                "event": "scaf_leak_monitor",
                "step": step,
                "verdict": "ERROR",
                "error": f"{type(exc).__name__}: {exc}",
            }
            self.last_scorecard = None
        finally:
            self._last_step = step
            if want_honest:
                self._last_honest_step = step
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        record["elapsed_s"] = round(time.time() - t0, 3)
        record["aile_delta"] = self._delta(record.get("aile"))
        self.history.append(record)
        self._write(record)

        if self.raise_on_leak and record["verdict"] == "LEAK":
            raise CausalLeakError(
                f"SCAF LeakMonitor detected a leak at step {step}\n"
                + (
                    self.last_scorecard.summary()
                    if self.last_scorecard
                    else json.dumps(record, indent=2)
                )
            )
        return record

    # ------------------------------------------------------------------
    def _measure(self, step: int, want_honest: bool) -> dict[str, Any]:
        # The corpus is rewound so every run probes identical sequences.
        # Without this, a change in AILE between two steps could be a change
        # in the draw rather than in the model, and the trend would be noise.
        self.corpus.reset()

        # Hooks are installed here and removed in `close`, rather than being
        # held for the lifetime of the monitor. A hook left registered runs on
        # every training forward pass, which is both a cost and a way for the
        # auditor to perturb what it audits.
        with preserved_rng(), InterventableModel(
            self.model,
            adapter=self.adapter,
            device=self.device,
            dtype=self.dtype,
        ) as im:
            controls = []
            if self.controls:
                n = min(self.n_seqs, 4)
                controls = [
                    DeterminismControl(n_seqs=n).run(im, self.corpus),
                    PlaceboControl(n_seqs=n).run(im, self.corpus),
                    PositiveControl(n_seqs=n).run(im, self.corpus),
                ]

            det = controls[0] if controls else None
            floor = (
                det.statistic if det is not None and det.passed is False
                else 0.0
            )

            probes = [
                FuturePerturbationProbe(
                    splits=self.splits,
                    n_seqs=self.n_seqs,
                    n_pairs=self.n_pairs,
                    threshold=floor,
                    micro_batch=self.micro_batch,
                ).run(im, self.corpus)
            ]
            if want_honest:
                probes.append(
                    TargetRelocationProbe(
                        n_seqs=self.n_seqs,
                        n_targets=self.n_targets,
                        threshold=self.relocation_threshold,
                        micro_batch=self.micro_batch,
                    ).run(im, self.corpus)
                )

            card = LeakScorecard(
                model=type(im.model).__name__,
                adapter=im.adapter.name,
                dtype=str(im.dtype),
                device=str(im.device),
                controls=controls,
                probes=probes,
                config=im.config(),
                notes=(
                    ()
                    if self.controls
                    else (
                        "controls disabled: the verdict is INVALID by "
                        "construction and only the effect sizes are usable",
                    )
                ),
            )
            n_forwards = im.n_forwards

        self.last_scorecard = card
        fp, tr = probes[0], (probes[1] if len(probes) > 1 else None)

        record: dict[str, Any] = {
            "event": "scaf_leak_monitor",
            "step": step,
            "verdict": card.verdict,
            "linf": fp.statistic,
            "aile": fp.detail.get("aile"),
            "linf_threshold": fp.threshold,
            "n_forwards": n_forwards,
            "seq_len": self.corpus.seq_len,
            "n_seqs": self.n_seqs,
        }
        if controls:
            record["determinism_floor"] = controls[0].statistic
            record["placebo"] = controls[1].statistic
            record["positive_control"] = controls[2].statistic
            record["controls_ok"] = card.controls_ok
        if tr is not None and not tr.skipped:
            record.update(
                tau_leak=tr.statistic,
                nll_standard=tr.detail["nll_standard"],
                nll_honest=tr.detail["nll_honest"],
                ppl_standard=tr.detail["ppl_standard"],
                ppl_honest=tr.detail["ppl_honest"],
                ppl_inflation=tr.detail["ppl_inflation"],
                ppl_saturated=tr.detail["ppl_saturated"],
            )
        elif tr is not None:
            record["tau_leak_skipped"] = tr.skipped_reason
        return record

    # ------------------------------------------------------------------
    def _delta(self, aile) -> float | None:
        """Change in AILE since the previous successful run."""
        if aile is None:
            return None
        for r in reversed(self.history):
            prev = r.get("aile")
            if prev is not None:
                return aile - prev
        return None

    def _write(self, record: dict[str, Any]) -> None:
        if self.jsonl_path is None:
            return
        try:
            with open(self.jsonl_path, "a") as fh:
                fh.write(json.dumps(record, default=str) + "\n")
        except OSError as exc:  # noqa: BLE001 - a full disk must not stop training
            record.setdefault("warnings", []).append(f"jsonl write failed: {exc}")

    # ------------------------------------------------------------------
    def summary(self) -> str:
        """Human-readable history, for printing at the end of a run."""
        if not self.history:
            return "SCAF LeakMonitor — no measurements yet"
        lines = [
            f"SCAF LeakMonitor — {len(self.history)} measurements, "
            f"every {self.interval} steps",
            f"{'step':>9}  {'verdict':<8} {'linf':>11} {'AILE':>11} "
            f"{'tau_leak':>10}",
        ]
        for r in self.history:
            tau = r.get("tau_leak")
            lines.append(
                f"{r['step']:>9}  {r['verdict']:<8} "
                f"{_fmt(r.get('linf')):>11} {_fmt(r.get('aile')):>11} "
                f"{_fmt(tau, '{:.4f}'):>10}"
            )
        first = self.first_leak_step
        if first is not None:
            lines.append(
                f"  first leak detected at step {first}; every metric "
                "logged from that point on was measured through it"
            )
        return "\n".join(lines)

    def __str__(self) -> str:  # pragma: no cover - debugging aid
        return self.summary()

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"<LeakMonitor interval={self.interval} "
            f"runs={len(self.history)} model={type(self.model).__name__}>"
        )


# ----------------------------------------------------------------------
def _fmt(v, spec: str = "{:.3e}") -> str:
    return "-" if v is None else spec.format(v)


def _device_of(model) -> torch.device:
    return next(
        (p.device for p in model.parameters()), torch.device("cpu")
    )


def _build_corpus(model, tokens, seq_len, seed, vocab_size, adapter):
    if tokens is not None:
        return TokenCorpus(
            tokens, seq_len=seq_len, seed=seed, vocab_size=vocab_size
        )
    v = vocab_size
    if not v:
        from .core.adapters import resolve_adapter

        v = resolve_adapter(model, adapter).config(model).get("vocab_size")
    if not v:
        raise ValueError(
            "no tokens given and vocab_size could not be inferred from the "
            "model config; pass tokens= or vocab_size="
        )
    return SyntheticCorpus(v, seq_len=seq_len, seed=seed)
