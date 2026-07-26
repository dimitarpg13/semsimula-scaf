"""Shared test setup: import path and a deterministic starting point."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(autouse=True)
def _deterministic_toy_init():
    """Seed torch before every test so toy models initialise identically.

    The toys draw random embeddings at construction, and several assertions
    depend on quantities that vary with that draw — attribution shares in
    particular, where a weak second carrier can land just either side of zero
    depending on how the two channels happen to correlate. Left unseeded, the
    suite failed roughly one run in seven for reasons having nothing to do
    with the code under test. A framework that exists to make audits
    reproducible should start by making its own tests reproducible.
    """
    torch.manual_seed(0)
