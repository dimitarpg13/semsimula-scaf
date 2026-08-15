"""SCAF — SemSimula Causal Auditing Framework.

Detects, sizes, and attributes causal leaks in autoregressive sequence models.

A *causal leak* is any path by which a token at position ``s`` influences the
model's prediction at an earlier position ``t < s``. Such a path inflates
perplexity in a way that no held-out split can detect, because the leak is
present at evaluation time too. SCAF exists because that failure mode is
invisible to every conventional metric.

Quickstart::

    import scaf

    report = scaf.audit(model, tokens=val_tokens, device="cuda")
    print(report.summary())
    report.assert_causal()   # raises CausalLeakError on a leak

Inside a training loop, to record *when* a leak opens rather than only that it
did::

    monitor = scaf.LeakMonitor(model, tokens=val_tokens, interval=20_000)
    for step in range(total_steps):
        ...
        monitor.maybe_run(step)

And to hand a measured leak to a formal causal-inference engine::

    frame = scaf.build_leak_frame(model, tokens=val_tokens)
    print(frame.summary())                       # torch only
    print(scaf.estimate_leak(frame).summary())   # needs the [pywhy] extra
"""

from __future__ import annotations

from .api import audit
from .controls import DeterminismControl, PlaceboControl, PositiveControl
from .core.adapters import (
    Capabilities,
    FockAdapter,
    GenericAdapter,
    ModelAdapter,
    register_adapter,
    registered_adapters,
    resolve_adapter,
)
from .core.corpus import Corpus, SyntheticCorpus, TokenCorpus
from .core.intervenable import InterventableModel
from .estimate.frames import LeakFrame, build_leak_frame
from .estimate.pywhy import EstimationReport, RefutationResult, estimate_leak
from .monitor import LeakMonitor
from .probes.base import Probe, ProbeResult
from .probes.future_perturbation import FuturePerturbationProbe
from .probes.hidden_state import HiddenStateLeakProbe
from .probes.mediation import MediationProbe
from .probes.target_relocation import TargetRelocationProbe
from .report import CausalLeakError, LeakScorecard

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "audit",
    # core
    "InterventableModel",
    "Corpus",
    "TokenCorpus",
    "SyntheticCorpus",
    # adapters
    "ModelAdapter",
    "Capabilities",
    "FockAdapter",
    "GenericAdapter",
    "register_adapter",
    "registered_adapters",
    "resolve_adapter",
    # probes
    "Probe",
    "ProbeResult",
    "FuturePerturbationProbe",
    "HiddenStateLeakProbe",
    "TargetRelocationProbe",
    "MediationProbe",
    # controls
    "DeterminismControl",
    "PlaceboControl",
    "PositiveControl",
    # continuous monitoring
    "LeakMonitor",
    # formal estimands
    "LeakFrame",
    "build_leak_frame",
    "estimate_leak",
    "EstimationReport",
    "RefutationResult",
    # reporting
    "LeakScorecard",
    "CausalLeakError",
]
