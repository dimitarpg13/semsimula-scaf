"""Formal estimands for measured leaks.

Two layers, deliberately separated by dependency weight:

* :mod:`scaf.estimate.frames` is pure torch. It turns the future-perturbation
  intervention into a tidy interventional dataset and computes the exact
  paired ATE. Nothing here needs pandas, DoWhy, or EconML.
* :mod:`scaf.estimate.pywhy` wraps the PyWhy stack for refutation and
  heterogeneous effects. Its heavy imports happen inside
  :func:`~scaf.estimate.pywhy.estimate_leak`, so importing this package is
  always safe and a missing extra surfaces as a clear message at call time
  rather than an import error at notebook startup.
"""

from __future__ import annotations

from .frames import LeakFrame, build_leak_frame
from .pywhy import EstimationReport, RefutationResult, estimate_leak

__all__ = [
    "LeakFrame",
    "build_leak_frame",
    "EstimationReport",
    "RefutationResult",
    "estimate_leak",
]
