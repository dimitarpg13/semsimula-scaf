"""Integration against the real Fock-PARFLM models in the research repo.

Everything else in the suite runs against toys with stipulated structure. This
file runs against the actual model the framework was built for, and asserts the
one thing that matters: SCAF must independently reproduce the verdict of the
hand-written ``fock_causality_probe.py`` — a leak under the legacy register
implementation, and bit-exact zero under the prefix-causal fix.

Skipped unless the research repo is available. Point ``SEMSIMULA_PAPER`` at it,
or keep it as a sibling checkout of this repo.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
import torch

import scaf

pytestmark = pytest.mark.semsimula


def _repo() -> Path | None:
    env = os.environ.get("SEMSIMULA_PAPER")
    candidates = [Path(env)] if env else []
    candidates.append(Path(__file__).resolve().parents[2] / "semsimula-paper")
    for c in candidates:
        if (c / "notebooks" / "conservative_arch" / "parf").is_dir():
            return c
    return None


REPO = _repo()
pytestmark = [
    pytestmark,
    pytest.mark.skipif(
        REPO is None,
        reason="research repo not found; set SEMSIMULA_PAPER to enable",
    ),
]


@pytest.fixture(scope="module")
def fock():
    """Import the real multi-xi Fock model, or skip."""
    arch = REPO / "notebooks" / "conservative_arch"
    for p in (str(arch), str(arch / "parf")):
        if p not in sys.path:
            sys.path.insert(0, p)
    try:
        from parf.model_fock_parf_multixi import (  # noqa: PLC0415
            FockMultiXiPARFConfig,
            FockMultiXiPARFLM,
        )
    except ImportError as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"cannot import the Fock model: {exc}")
    return FockMultiXiPARFLM, FockMultiXiPARFConfig


# Small enough to run in seconds, large enough to exercise the real register
# machinery: multiple layers, multiple xi channels, and a live reverse channel.
BASE = dict(
    vocab_size=256, d=64, L=3, max_len=64, v_hidden=64, xi_channels=4,
    n_registers=4, fock_version="v2", reverse_channel=True,
    mass_mode="global",
)
PROBE = dict(vocab_size=256, seq_len=32, n_seqs=2, n_targets=4, micro_batch=1)


def _build(fock, *, prefix_causal, gate):
    FockLM, Config = fock
    torch.manual_seed(0)
    model = FockLM(Config(**BASE, prefix_causal_registers=prefix_causal)).eval()
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "reverse_channel_scale" in name:
                p.fill_(gate)
    return model


def test_fock_adapter_is_selected(fock):
    model = _build(fock, prefix_causal=True, gate=0.0)
    assert scaf.resolve_adapter(model).name == "fock"


def test_reverse_channel_starts_closed(fock):
    """The premise of continuous monitoring, verified on the real model.

    ``reverse_channel_scale`` is initialised to zero and the gate is
    ``tanh``, so the channel is shut at step zero and no probe can see it. The
    leak only becomes measurable once training opens the valve — which is why
    a single audit of the architecture at initialisation is not enough, and
    :class:`~scaf.LeakMonitor` has to run throughout the run.
    """
    model = _build(fock, prefix_causal=False, gate=0.0)
    scale = dict(model.named_parameters())["reverse_channel_scale"]
    assert float(scale.reshape(-1)[0]) == 0.0

    record = scaf.LeakMonitor(model, interval=1, **PROBE).run(0)
    assert record["linf"] == 0.0, "a closed gate must leak nothing"


def test_legacy_registers_leak_once_the_gate_opens(fock):
    """Reproduces the L3 reverse-channel finding on the real model."""
    model = _build(fock, prefix_causal=False, gate=2.0)
    record = scaf.LeakMonitor(model, interval=1, **PROBE).run(0)
    assert record["verdict"] == "LEAK"
    assert record["linf"] > 0.0
    assert record["aile"] > 0.0


def test_prefix_causal_registers_are_bit_exactly_clean(fock):
    """The fix, verified independently of the hand-written probe.

    Same architecture, same open gate, same probe — the only change is the
    register lifecycle. Zero here is bit-exact rather than merely small,
    because a structurally causal prefix computation cannot be touched by
    suffix values at all.
    """
    model = _build(fock, prefix_causal=True, gate=2.0)
    record = scaf.LeakMonitor(model, interval=1, **PROBE).run(0)
    assert record["linf"] == 0.0
    assert record["aile"] == 0.0
    assert record["verdict"] == "CLEAN"


def test_the_fix_is_what_changes_the_verdict(fock):
    """Paired comparison, so nothing but the register flag can explain it."""
    leaky = scaf.LeakMonitor(
        _build(fock, prefix_causal=False, gate=2.0), interval=1, **PROBE
    ).run(0)
    fixed = scaf.LeakMonitor(
        _build(fock, prefix_causal=True, gate=2.0), interval=1, **PROBE
    ).run(0)
    assert leaky["linf"] > fixed["linf"] == 0.0
    assert (leaky["verdict"], fixed["verdict"]) == ("LEAK", "CLEAN")


def test_audit_and_frame_agree_with_the_monitor(fock):
    """The three entry points must not tell different stories."""
    model = _build(fock, prefix_causal=False, gate=2.0)
    report = scaf.audit(model, mediation=False, **PROBE)
    frame = scaf.build_leak_frame(
        model, seq_len=32, n_seqs=2, n_pairs=1, max_positions=4,
        micro_batch=1, vocab_size=256,
    )
    assert report.verdict == "LEAK"
    assert report.get("future_perturbation").statistic > 0.0
    assert frame.ate() != 0.0
