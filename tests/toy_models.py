"""Toy models with known causal structure — SCAF's ground truth.

A causal auditor that is only ever tested against models whose true structure
is unknown cannot be validated. These four toys have structure we *stipulate*,
so every assertion in the suite is checkable:

===========================  ================================================
Model                        Ground truth
===========================  ================================================
:class:`CausalToyLM`         strictly causal; every probe must report zero
:class:`LeakyToyLM`          global mean pool; future reaches past
:class:`PeekingToyLM`        copies the next token; huge honest-PPL gap
:class:`DeafToyLM`           ignores its input; probes cannot see anything
===========================  ================================================

:class:`PeekingToyLM` is the miniature of the register leak: standard PPL near
1, honest PPL near vocabulary size. :class:`DeafToyLM` is the miniature of the
*blind probe* — it must produce ``INVALID``, never ``CLEAN``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

__all__ = [
    "ToyConfig",
    "CausalToyLM",
    "LeakyToyLM",
    "PeekingToyLM",
    "PureLookaheadToyLM",
    "DeafToyLM",
    "FockLikeToyLM",
]


@dataclass
class ToyConfig:
    vocab_size: int = 24
    d: int = 16
    L: int = 1
    max_len: int = 64
    causal_force: bool = False
    prefix_causal_registers: bool = True


class _ToyBase(nn.Module):
    """Shared scaffolding: tied embedding/unembedding for predictable logits."""

    def __init__(self, cfg: ToyConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or ToyConfig()
        self.emb = nn.Embedding(self.cfg.vocab_size, self.cfg.d)
        self.out = nn.Linear(self.cfg.d, self.cfg.vocab_size, bias=False)
        # Tying makes out(emb(tok)) peak at tok, so a model that routes token
        # k into position t provably predicts k there. That determinism is what
        # lets the tests assert on PPL rather than on a fuzzy inequality.
        self.out.weight = self.emb.weight

    @staticmethod
    def _prefix_mean(h: torch.Tensor) -> torch.Tensor:
        T = h.shape[1]
        denom = torch.arange(
            1, T + 1, device=h.device, dtype=h.dtype
        ).view(1, T, 1)
        return h.cumsum(dim=1) / denom


class CausalToyLM(_ToyBase):
    """Strictly causal: position ``t`` reads a mean over ``0..t`` only."""

    def forward(self, x):
        h = self.emb(x)
        return self.out(self._prefix_mean(h)), None


class LeakyToyLM(_ToyBase):
    """Adds a mean over *all* positions — a miniature shared register state.

    Structurally identical to the pre-fix Fock register: a single global
    summary, written by every token, readable at every position.
    """

    def __init__(self, cfg: ToyConfig | None = None, leak_scale: float = 1.0):
        super().__init__(cfg)
        self.leak_scale = float(leak_scale)

    def forward(self, x):
        h = self.emb(x)
        global_pool = h.mean(dim=1, keepdim=True)
        return self.out(self._prefix_mean(h) + self.leak_scale * global_pool), None


class PeekingToyLM(_ToyBase):
    """Legitimate causal processing *plus* a peek at the next token.

    With tied weights the peeked term dominates, driving standard PPL to nearly
    1 while honest PPL stays near chance — the miniature of the 7.69-vs-258.07
    split the d=384 checkpoint showed.

    The prefix-mean term is not decoration. A purely anti-causal model never
    reads its own position, so :class:`~scaf.PositiveControl` correctly refuses
    to certify the audit and the verdict comes back ``INVALID`` rather than
    ``LEAK``. Real leaky models are never purely anti-causal: they do genuine
    causal work and leak *in addition*. Keeping both terms makes this toy an
    honest stand-in.
    """

    def __init__(self, cfg: ToyConfig | None = None, gain: float = 8.0):
        super().__init__(cfg)
        self.gain = float(gain)

    def forward(self, x):
        h = self.emb(x)
        # roll(-1) moves position t+1's embedding to position t. The final
        # position wraps around to position 0, which is harmless: probes never
        # score the last position because it has no target.
        peeked = torch.roll(h, shifts=-1, dims=1)
        return self.out(self._prefix_mean(h) + self.gain * peeked), None


class PureLookaheadToyLM(_ToyBase):
    """Reads *only* the next token — never its own position.

    Pathological on purpose. Because position ``t`` does not depend on ``x_t``,
    the positive control cannot elicit a response in the causal direction, so
    SCAF must return ``INVALID``. Reporting ``LEAK`` here would overstate what
    the controls actually established.
    """

    def __init__(self, cfg: ToyConfig | None = None, gain: float = 8.0):
        super().__init__(cfg)
        self.gain = float(gain)

    def forward(self, x):
        peeked = torch.roll(self.emb(x), shifts=-1, dims=1)
        return self.out(self.gain * peeked), None


class DeafToyLM(_ToyBase):
    """Ignores its input entirely — the blind-probe stand-in.

    Every intervention produces zero response, so the leak probes trivially
    "pass". SCAF must refuse to certify this: the positive control has to fail
    and the verdict has to be ``INVALID``.
    """

    def forward(self, x):
        B, T = x.shape
        h = self.emb.weight[0].view(1, 1, -1).expand(B, T, -1)
        return self.out(h), None


class FockLikeToyLM(LeakyToyLM):
    """Carries Fock attribute names so :class:`~scaf.FockAdapter` detects it.

    Used to test adapter resolution and mediator knockout without needing the
    research repo on ``sys.path``.
    """

    def __init__(self, cfg: ToyConfig | None = None, leak_scale: float = 1.0):
        super().__init__(cfg, leak_scale=leak_scale)
        self.register_embed = nn.Parameter(torch.zeros(4, self.cfg.d))
        self.creation_gate = nn.Identity()
        self.reverse_ch = nn.Identity()
        self.reverse_channel_scale = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        h = self.emb(x)
        global_pool = h.mean(dim=1, keepdim=True)
        gate = torch.sigmoid(self.reverse_channel_scale).view(1, 1, 1)
        # The leak is carried *only* through the gate, so clamping the gate to
        # a large negative value must restore exact causality — which is what
        # the mediation knockout test asserts.
        return self.out(
            self._prefix_mean(h) + self.leak_scale * gate * global_pool
        ), None
