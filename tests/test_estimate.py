"""Frames, exact paired inference, and the DoWhy/EconML bridge.

The toys have stipulated causal structure, so every number here is checkable
against ground truth rather than against a previous run:

* :class:`~tests.toy_models.CausalToyLM` — ATE exactly zero, p-value 1.
* :class:`~tests.toy_models.PeekingToyLM` — reads only ``x_{t+1}``, so the
  entire effect must land in the ``distance_to_cut == 0`` stratum and be
  exactly zero everywhere else. That is the sharpest available check on the
  heterogeneity machinery: a profile that smears the effect across distances
  is wrong, no matter how plausible it looks.
"""

from __future__ import annotations

import csv
import importlib.util

import numpy as np
import pytest
import torch

import scaf
from scaf.estimate import build_leak_frame
from tests.toy_models import (
    CausalToyLM,
    LeakyToyLM,
    PeekingToyLM,
    ToyConfig,
)

CFG = ToyConfig(vocab_size=24, d=16, max_len=64)
KW = dict(
    vocab_size=24, seq_len=32, n_seqs=4, n_pairs=2, max_positions=6,
    micro_batch=0,
)

HAS_DOWHY = importlib.util.find_spec("dowhy") is not None
HAS_PANDAS = importlib.util.find_spec("pandas") is not None
HAS_ECONML = importlib.util.find_spec("econml") is not None

needs_dowhy = pytest.mark.skipif(
    not HAS_DOWHY, reason="requires the [pywhy] extra"
)
needs_pandas = pytest.mark.skipif(not HAS_PANDAS, reason="requires pandas")


@pytest.fixture
def causal_frame():
    return build_leak_frame(CausalToyLM(CFG), **KW)


@pytest.fixture
def peeking_frame():
    torch.manual_seed(0)
    return build_leak_frame(PeekingToyLM(CFG), **KW)


# ----------------------------------------------------------------------
# Frame construction
# ----------------------------------------------------------------------
def test_frame_is_balanced_and_paired(causal_frame):
    f = causal_frame
    arms = f.column("future_perturbed")
    assert sum(arms) * 2 == len(f), "each unit must contribute both arms"
    assert f.n_units * 2 == len(f)
    assert all(len(v) == len(f) for v in f.columns.values())


def test_frame_records_the_diagnostic_columns(causal_frame):
    for name in (
        "unit_id", "future_perturbed", "nll", "nll_within", "logit_l1",
        "position", "distance_to_cut", "split_frac", "target_perturbed",
    ):
        assert name in causal_frame.columns, name


def test_only_the_position_at_the_cut_has_a_resampled_target(causal_frame):
    """``target_perturbed`` must mean exactly ``t == t_p``."""
    f = causal_frame
    for d, tp in zip(
        f.column("distance_to_cut"),
        f.column("target_perturbed"),
        strict=True,
    ):
        assert tp == int(d == 0)


def test_causal_model_has_bit_exact_zero_effect(causal_frame):
    assert causal_frame.ate() == 0.0
    assert causal_frame.naive_ate() == 0.0
    assert all(d == 0.0 for d in causal_frame.deltas())


def test_leaky_model_has_a_measurable_effect():
    torch.manual_seed(0)
    f = build_leak_frame(LeakyToyLM(CFG, leak_scale=2.0), **KW)
    assert abs(f.ate()) > 1e-6


def test_paired_and_naive_ate_agree_on_a_balanced_frame(peeking_frame):
    assert peeking_frame.ate() == pytest.approx(
        peeking_frame.naive_ate(), rel=1e-9
    )


def test_within_unit_centring_preserves_the_ate(peeking_frame):
    """The whole justification for the centred outcome.

    If centring moved the point estimate it would be a different estimand, not
    a variance reduction, and using it for inference would be indefensible.
    """
    f = peeking_frame
    assert f.ate("nll_within") == pytest.approx(f.ate("nll"), rel=1e-12)
    # Each pair is now symmetric about its own mean.
    assert sum(f.column("nll_within")) == pytest.approx(0.0, abs=1e-9)


def test_frame_accepts_real_tokens():
    tokens = np.random.default_rng(0).integers(0, CFG.vocab_size, size=4096)
    f = build_leak_frame(CausalToyLM(CFG), tokens=tokens, **{
        k: v for k, v in KW.items() if k != "vocab_size"
    })
    assert f.metadata["corpus"] == "TokenCorpus"
    assert f.ate() == 0.0


def test_frame_restores_model_dtype():
    model = CausalToyLM(CFG)
    before = next(model.parameters()).dtype
    build_leak_frame(model, dtype="float64", **KW)
    assert next(model.parameters()).dtype == before


def test_frame_refuses_when_no_position_is_scoreable():
    with pytest.raises(ValueError, match="warm-up"):
        build_leak_frame(
            CausalToyLM(CFG), vocab_size=24, seq_len=8, n_seqs=2,
            splits=(0.05,), first_frac=0.9, micro_batch=0,
        )


# ----------------------------------------------------------------------
# Heterogeneity, model-free
# ----------------------------------------------------------------------
def test_peeking_leak_is_confined_to_the_position_at_the_cut(peeking_frame):
    """Ground truth: this model reads ``x_{t+1}`` and nothing else.

    So perturbing the future can only matter where the resampled token is the
    scored target — distance zero. Any other stratum showing an effect would
    mean the frame is attributing the leak to positions that cannot carry it.
    """
    profile = peeking_frame.ate_by("distance_to_cut")
    assert profile[0] > 1.0
    assert all(v == 0.0 for d, v in profile.items() if d != 0)


def test_strata_are_ordered_numerically(peeking_frame):
    keys = list(peeking_frame.ate_by("distance_to_cut"))
    assert keys == sorted(keys)


def test_strata_counts_cover_every_pair(peeking_frame):
    counts = peeking_frame.strata_counts("distance_to_cut")
    assert sum(counts.values()) == peeking_frame.n_units
    assert set(counts) == set(peeking_frame.ate_by("distance_to_cut"))


# ----------------------------------------------------------------------
# Exact paired inference
# ----------------------------------------------------------------------
def test_sign_flip_test_finds_a_real_leak(peeking_frame):
    r = peeking_frame.ate_test(n_permutations=2000, n_bootstrap=500)
    assert r["p_value"] < 0.01
    assert r["ci"][0] > 0.0, "interval must exclude zero"
    assert r["ate"] == pytest.approx(peeking_frame.ate())


def test_sign_flip_test_reports_no_effect_for_a_causal_model(causal_frame):
    r = causal_frame.ate_test(n_permutations=500, n_bootstrap=200)
    assert r["p_value"] == 1.0
    assert r["ci"] == (0.0, 0.0)


def test_sign_flip_p_value_is_reproducible(peeking_frame):
    a = peeking_frame.ate_test(n_permutations=500, n_bootstrap=0, seed=7)
    b = peeking_frame.ate_test(n_permutations=500, n_bootstrap=0, seed=7)
    assert a["p_value"] == b["p_value"]


def test_p_value_is_never_exactly_zero(peeking_frame):
    """Finite resolution must be reported honestly, not rounded to certainty."""
    r = peeking_frame.ate_test(n_permutations=100, n_bootstrap=0)
    assert r["p_value"] >= 1 / 101


def test_centring_removes_between_position_variance():
    """The precise claim behind the centred outcome.

    Raw NLL varies by nats across positions purely because an early target is
    harder than a late one, and none of that spread is caused by the
    treatment. Centring within a pair strips it while leaving the ATE
    untouched, so any estimator working on the centred column sees only the
    variance the treatment actually explains.

    The size of the win depends on how much between-position variance a model
    has, so this is measured on a model with a realistic NLL profile rather
    than on the peeking toy, whose factual NLL is near zero everywhere and
    therefore has little spread to remove.
    """
    torch.manual_seed(0)
    f = build_leak_frame(LeakyToyLM(CFG, leak_scale=0.5), **KW)
    raw = torch.tensor(f.column("nll")).std().item()
    centred = torch.tensor(f.column("nll_within")).std().item()
    assert centred < 0.5 * raw
    assert f.ate("nll_within") == pytest.approx(f.ate("nll"), rel=1e-12)


# ----------------------------------------------------------------------
# Graph and export
# ----------------------------------------------------------------------
def test_graph_has_no_edge_into_the_treatment(causal_frame):
    """Treatment is assigned by the auditor, so nothing may cause it.

    An edge into the treatment would assert confounding we know is absent, and
    would send DoWhy looking for a backdoor set that should be empty.
    """
    dot = causal_frame.to_dot()
    assert "future_perturbed -> nll;" in dot
    assert "-> future_perturbed;" not in dot


def test_graph_can_name_a_derived_outcome(causal_frame):
    dot = causal_frame.to_dot("nll_within")
    assert "future_perturbed -> nll_within;" in dot
    assert "position -> nll_within;" in dot


def test_csv_export_round_trips(causal_frame, tmp_path):
    path = tmp_path / "frame.csv"
    causal_frame.to_csv(path)
    with open(path) as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == len(causal_frame)
    assert set(rows[0]) == set(causal_frame.columns)


@needs_pandas
def test_pandas_export_matches_the_frame(causal_frame):
    df = causal_frame.to_pandas()
    assert len(df) == len(causal_frame)
    assert set(df.columns) == set(causal_frame.columns)


def test_unknown_column_raises_with_a_listing(causal_frame):
    with pytest.raises(KeyError, match="available"):
        causal_frame.column("no_such_column")


# ----------------------------------------------------------------------
# The PyWhy bridge
# ----------------------------------------------------------------------
@pytest.mark.skipif(HAS_DOWHY, reason="tests the missing-extra path")
def test_missing_extra_gives_an_actionable_message(causal_frame):
    with pytest.raises(ImportError, match=r"semsimula-scaf\[pywhy\]"):
        scaf.estimate_leak(causal_frame)


@needs_dowhy
@pytest.mark.slow
def test_dowhy_reproduces_the_exact_paired_ate(peeking_frame):
    """The bridge's own correctness check.

    Treatment is randomised, so an unbiased estimator has no freedom to
    disagree with the paired difference. If it does, the estimator is
    misconfigured — which is exactly what this assertion is for.
    """
    rep = scaf.estimate_leak(
        peeking_frame, cate_model=None, num_simulations=5
    )
    assert rep.ate == pytest.approx(peeking_frame.ate(), rel=1e-6)
    assert rep.agrees_with_reference


@needs_dowhy
@pytest.mark.slow
def test_refutations_pass_on_a_real_leak(peeking_frame):
    rep = scaf.estimate_leak(
        peeking_frame, cate_model=None, num_simulations=5
    )
    assert rep.refutations_ok
    placebo = next(
        r for r in rep.refutations if r.name == "placebo_treatment_refuter"
    )
    # Permuting the arm labels destroys the pairing, so the effect must vanish.
    assert abs(placebo.new_effect) < abs(placebo.original_effect)


@needs_dowhy
@pytest.mark.slow
def test_causal_model_estimates_to_zero(causal_frame):
    rep = scaf.estimate_leak(causal_frame, cate_model=None, num_simulations=5)
    assert rep.ate == pytest.approx(0.0, abs=1e-12)
    assert rep.exact_p_value == 1.0


@needs_dowhy
@pytest.mark.slow
def test_exact_profile_is_reported_even_without_econml(causal_frame):
    rep = scaf.estimate_leak(causal_frame, cate_model=None, refuters=())
    assert rep.cate["distance_to_cut"]["exact"]
    assert "fit" not in rep.cate["distance_to_cut"]


@needs_dowhy
@pytest.mark.skipif(not HAS_ECONML, reason="requires econml")
@pytest.mark.slow
def test_causal_forest_recovers_the_spike(peeking_frame):
    """The forest must find the one stratum that carries the leak.

    A linear CATE cannot: fitted to this profile it reports roughly a third of
    the true peak and a spurious negative effect far from the cut, which would
    read as the future *suppressing* the past.
    """
    rep = scaf.estimate_leak(peeking_frame, refuters=())
    fit = dict(rep.cate["distance_to_cut"]["fit"]["points"])
    exact = peeking_frame.ate_by("distance_to_cut")
    assert fit[0.0] == pytest.approx(exact[0], rel=0.1)
    assert all(abs(v) < 0.05 * exact[0] for k, v in fit.items() if k != 0.0)


@needs_dowhy
@pytest.mark.slow
def test_report_serialises_for_a_log(peeking_frame):
    rep = scaf.estimate_leak(
        peeking_frame, cate_model=None, num_simulations=5
    )
    d = rep.to_dict()
    assert d["ate"] == pytest.approx(rep.ate)
    assert d["exact_p_value"] == rep.exact_p_value
    assert isinstance(d["refutations"], list)
    assert "ATE" in rep.summary()
