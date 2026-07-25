"""Token corpora and the perturbation operations probes are built from.

A probe needs three things from a corpus: sequences to run, a way to replace a
*suffix* with a plausible alternative (the future-perturbation intervention),
and a way to replace a *single past position* (the positive control). Putting
all three here keeps the token distribution in one place — a perturbation drawn
from the wrong distribution would make a leak look larger or smaller than it is.

Two implementations:

* :class:`TokenCorpus` wraps real tokens, so perturbations are in-distribution.
  This is what you want for trained-scale audits.
* :class:`SyntheticCorpus` draws uniformly from the vocabulary and needs no data
  at all. Perfect for architectural probes, where the question is whether a
  future token can influence a past logit *at all* — a question about the
  computation graph, not about the data.
"""

from __future__ import annotations

import torch

__all__ = ["Corpus", "TokenCorpus", "SyntheticCorpus"]


def to_long_tensor(x) -> torch.Tensor:
    """Convert tokens from any common container to a 1-D CPU long tensor.

    Deliberately avoids ``torch.from_numpy`` and ``Tensor.numpy()``. Those go
    through torch's compiled NumPy bridge, which breaks outright when torch was
    built against a different NumPy major version than the one installed — a
    combination users hit regularly on Colab and older macOS wheels. Going via
    ``torch.as_tensor`` with an explicit dtype uses the buffer protocol
    instead, which keeps working.
    """
    if isinstance(x, torch.Tensor):
        return x.detach().to(device="cpu", dtype=torch.long).reshape(-1)
    try:
        return torch.as_tensor(x, dtype=torch.long).reshape(-1)
    except (TypeError, RuntimeError):
        # Last resort for exotic array types: go through Python ints.
        return torch.as_tensor(list(x), dtype=torch.long).reshape(-1)


class Corpus:
    """Base corpus: samples sequences and perturbs them.

    Args:
        seq_len: Length of sequences handed to probes.
        seed: Seed for the corpus's private RNG. Probes must be reproducible;
            using a private generator rather than global torch/numpy state
            keeps an audit deterministic regardless of what the caller's
            training loop has done to the global seed.
    """

    def __init__(self, seq_len: int = 128, seed: int = 0) -> None:
        if seq_len < 4:
            raise ValueError(f"seq_len must be >= 4, got {seq_len}")
        self.seq_len = int(seq_len)
        self.seed = int(seed)
        self._gen = torch.Generator().manual_seed(seed)

    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Rewind the RNG so a probe can be re-run identically."""
        self._gen = torch.Generator().manual_seed(self.seed)

    def sample(self, n: int) -> torch.Tensor:
        """Return ``n`` sequences, shape ``(n, seq_len)``, dtype long."""
        raise NotImplementedError

    def _random_tokens(self, shape: tuple[int, ...]) -> torch.Tensor:
        raise NotImplementedError

    def _randint(self, high: int, shape: tuple[int, ...]) -> torch.Tensor:
        return torch.randint(
            0, high, shape, generator=self._gen, dtype=torch.long
        )

    # ------------------------------------------------------------------
    # Interventions
    # ------------------------------------------------------------------
    def perturb_suffix(self, x: torch.Tensor, t_p: int) -> torch.Tensor:
        """``do(x_{>t_p} := x~_{>t_p})`` — replace the future, keep the prefix.

        This is the core intervention of the audit. In a correctly causal
        autoregressive model the returned sequence must produce *identical*
        logits at every position ``<= t_p``.

        Args:
            x: ``(B, T)`` token batch.
            t_p: Last position of the preserved prefix. Positions
                ``t_p + 1 ... T-1`` are resampled.

        Returns:
            A new tensor; ``x`` is not modified.
        """
        B, T = self._check(x)
        if not 0 <= t_p < T - 1:
            raise ValueError(
                f"t_p must satisfy 0 <= t_p < T-1 = {T - 1}, got {t_p}"
            )
        out = x.clone()
        n_future = T - (t_p + 1)
        out[:, t_p + 1:] = self._random_tokens((B, n_future)).to(x.device)
        return out

    def perturb_position(self, x: torch.Tensor, s: int) -> torch.Tensor:
        """Replace exactly one position, guaranteeing the token changes.

        Used by the positive control: perturbing a position the model *is*
        allowed to see must produce a visible effect. If it does not, the probe
        is not actually reaching the model and every "no leak" result it
        produces is worthless. This is the direct guard against the blind-probe
        failure that let the register leak go undetected.
        """
        B, T = self._check(x)
        if not 0 <= s < T:
            raise ValueError(f"s must satisfy 0 <= s < {T}, got {s}")
        out = x.clone()
        for b in range(B):
            original = int(out[b, s])
            for _ in range(32):
                candidate = int(self._random_tokens((1,))[0])
                if candidate != original:
                    out[b, s] = candidate
                    break
            else:
                # Degenerate vocabulary (size 1); the caller's positive control
                # will fail loudly, which is the correct outcome.
                pass
        return out

    @staticmethod
    def _check(x: torch.Tensor) -> tuple[int, int]:
        if x.dim() != 2:
            raise ValueError(f"expected a (B, T) token batch, got {tuple(x.shape)}")
        return int(x.shape[0]), int(x.shape[1])


class SyntheticCorpus(Corpus):
    """Uniform random tokens. No data required.

    Adequate for architectural probes: whether position ``t`` can *reach*
    position ``s > t`` through the computation graph does not depend on the
    tokens being realistic. Not adequate for trained-scale leak sizing, where
    an out-of-distribution suffix would understate how much usable information
    a real leak carries.
    """

    def __init__(
        self, vocab_size: int, seq_len: int = 128, seed: int = 0
    ) -> None:
        super().__init__(seq_len=seq_len, seed=seed)
        if vocab_size < 2:
            raise ValueError(f"vocab_size must be >= 2, got {vocab_size}")
        self.vocab_size = int(vocab_size)

    def _random_tokens(self, shape):
        return self._randint(self.vocab_size, shape)

    def sample(self, n: int) -> torch.Tensor:
        return self._random_tokens((n, self.seq_len))

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"<SyntheticCorpus vocab={self.vocab_size} "
            f"seq_len={self.seq_len} seed={self.seed}>"
        )


class TokenCorpus(Corpus):
    """Real tokens — a flat id stream or a set of pre-chunked sequences.

    Args:
        tokens: 1-D stream of token ids, or a 2-D ``(N, T)`` array of
            sequences. A 2-D array is flattened; chunking is done here so that
            ``seq_len`` is honoured regardless of how the caller chunked.
        seq_len: Length of sequences handed to probes.
        seed: RNG seed for window selection and perturbation.
        vocab_size: Optional true vocabulary size. When omitted it is inferred
            as ``max(tokens) + 1``, which understates the vocabulary if rare
            ids are absent — harmless for perturbation, since resampling from
            observed tokens is in-distribution by construction.
    """

    def __init__(
        self,
        tokens,
        seq_len: int = 128,
        seed: int = 0,
        vocab_size: int | None = None,
    ) -> None:
        super().__init__(seq_len=seq_len, seed=seed)
        arr = to_long_tensor(tokens)
        if arr.numel() < seq_len + 1:
            raise ValueError(
                f"corpus has {arr.numel()} tokens, need at least seq_len+1 = "
                f"{seq_len + 1}"
            )
        self.tokens = arr
        self.vocab_size = (
            int(vocab_size) if vocab_size else int(arr.max()) + 1
        )

    def _random_tokens(self, shape):
        """Resample from the *empirical* token distribution.

        Drawing from observed tokens rather than uniformly over the vocabulary
        keeps the counterfactual future in-distribution, so a measured leak
        reflects information the model could actually exploit at inference.
        """
        idx = self._randint(self.tokens.numel(), shape)
        return self.tokens[idx]

    def sample(self, n: int) -> torch.Tensor:
        starts = self._randint(self.tokens.numel() - self.seq_len, (n,))
        offsets = torch.arange(self.seq_len, dtype=torch.long)
        return self.tokens[starts[:, None] + offsets[None, :]]

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"<TokenCorpus n_tokens={self.tokens.numel()} "
            f"vocab={self.vocab_size} seq_len={self.seq_len}>"
        )
