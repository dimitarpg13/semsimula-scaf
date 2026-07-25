"""The one-call entry point: :func:`audit`.

Designed so a notebook cell that already has a model can add causal auditing in
one line, with no restructuring::

    report = scaf.audit(model, tokens=val_tokens, device="cuda")
    print(report.summary())
"""

from __future__ import annotations

import torch
from torch import nn

from .controls import DeterminismControl, PlaceboControl, PositiveControl
from .core.corpus import Corpus, SyntheticCorpus, TokenCorpus
from .core.intervenable import InterventableModel
from .probes.future_perturbation import FuturePerturbationProbe
from .probes.target_relocation import TargetRelocationProbe
from .report import LeakScorecard

__all__ = ["audit"]

_DTYPES = {
    "float64": torch.float64, "f64": torch.float64, "double": torch.float64,
    "float32": torch.float32, "f32": torch.float32, "float": torch.float32,
}


def _resolve_dtype(dtype):
    if dtype is None or isinstance(dtype, torch.dtype):
        return dtype
    key = str(dtype).lower().replace("torch.", "")
    if key not in _DTYPES:
        raise ValueError(
            f"unknown dtype {dtype!r}; use one of {sorted(_DTYPES)}"
        )
    return _DTYPES[key]


def audit(
    model: nn.Module | InterventableModel,
    tokens=None,
    *,
    corpus: Corpus | None = None,
    adapter=None,
    device: str | torch.device = "cpu",
    dtype=None,
    seq_len: int = 128,
    n_seqs: int = 8,
    n_targets: int = 32,
    micro_batch: int = 4,
    seed: int = 0,
    vocab_size: int | None = None,
    relocation_threshold: float = 1e-3,
) -> LeakScorecard:
    """Run the causal-leak audit battery and return a scorecard.

    Args:
        model: The model under test, or an existing
            :class:`~scaf.core.intervenable.InterventableModel`.
        tokens: Real token ids (1-D stream or 2-D batch). Strongly preferred
            for trained checkpoints: a leak's *size* depends on the future
            being in-distribution. Omit only for architectural probes, where a
            synthetic corpus is built automatically.
        corpus: Explicit corpus, overriding ``tokens``.
        adapter: Explicit adapter; auto-detected when omitted.
        device: Device to run on.
        dtype: ``"float64"`` for architectural probes on small models (gives a
            bit-exact zero baseline); ``"float32"`` for real checkpoints.
            ``None`` leaves the model as-is.
        seq_len: Probe sequence length.
        n_seqs: Sequences per probe.
        n_targets: Target positions for the honest-PPL probe. This dominates
            runtime — each one costs a forward pass.
        micro_batch: Forward chunk size, to bound peak memory.
        seed: Corpus seed.
        vocab_size: Needed only when no tokens are given and the adapter cannot
            read a vocabulary size from the model config.
        relocation_threshold: Tolerated honest-PPL gap in nats.

    Returns:
        A :class:`~scaf.report.LeakScorecard`. Check ``.verdict`` or call
        ``.assert_causal()``.
    """
    owns = not isinstance(model, InterventableModel)
    im = (
        InterventableModel(
            model, adapter=adapter, device=device, dtype=_resolve_dtype(dtype)
        )
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

        max_len = cfg.get("max_len")
        notes = list(im.caps.notes)
        if max_len and corpus.seq_len > max_len:
            raise ValueError(
                f"seq_len={corpus.seq_len} exceeds the model's max_len={max_len}"
            )
        if tokens is None:
            notes.append(
                "synthetic corpus: leak DETECTION is valid, but leak SIZE is "
                "not — pass real tokens to size a leak in nats"
            )

        controls = [
            DeterminismControl(n_seqs=min(n_seqs, 4)).run(im, corpus),
            PlaceboControl(n_seqs=min(n_seqs, 4)).run(im, corpus),
            PositiveControl(n_seqs=min(n_seqs, 4)).run(im, corpus),
        ]

        # The determinism control measures the platform's reproducibility
        # floor. A leak smaller than that floor is unmeasurable, so the
        # threshold is raised to it rather than left at a zero we cannot
        # actually resolve.
        det = controls[0]
        floor = det.statistic if det.passed is False else 0.0
        if floor > 0:
            notes.append(
                f"non-deterministic forward (floor {floor:.3g}); leak "
                "thresholds raised accordingly and the audit is INVALID"
            )

        probes = [
            FuturePerturbationProbe(
                n_seqs=n_seqs, threshold=floor, micro_batch=micro_batch
            ).run(im, corpus),
            TargetRelocationProbe(
                n_seqs=n_seqs,
                n_targets=n_targets,
                threshold=relocation_threshold,
                micro_batch=micro_batch,
            ).run(im, corpus),
        ]

        return LeakScorecard(
            model=type(im.model).__name__,
            adapter=im.adapter.name,
            dtype=str(im.dtype),
            device=str(im.device),
            controls=controls,
            probes=probes,
            config=cfg,
            notes=tuple(notes),
        )
    finally:
        if owns:
            im.close()
