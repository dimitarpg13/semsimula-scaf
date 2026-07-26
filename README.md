# SCAF — SemSimula Causal Auditing Framework

Detects, sizes, and attributes **causal leaks** in autoregressive sequence models.

A causal leak is any path by which a token at position `s` influences the model's
prediction at an earlier position `t < s`. Leaks are invisible to every
conventional metric: a held-out split does not catch them, because the leak is
present at evaluation time too. The model simply reports a perplexity it did not
earn.

SCAF exists because this happened. A Fock-PARFLM checkpoint reported **7.69
perplexity**; its honest, leak-free perplexity was **258.07** — a gap of
**+3.51 nats**, roughly 33x inflation. The leak had survived a full causal audit,
because the probe used to certify it was not actually exercising the leaking
pathway.

## Install

```bash
pip install git+https://github.com/dimitarpg13/semsimula-scaf
```

Optional extras: `[pywhy]` for DoWhy/EconML estimands, `[plot]` for figures,
`[dev]` for the test suite.

## Quickstart

```python
import scaf

report = scaf.audit(model, tokens=val_tokens, device="cuda", dtype="float32")
print(report.summary())
report.assert_causal()      # raises CausalLeakError unless the verdict is CLEAN
```

```
SCAF audit — FockMultiXiPARFLM
  adapter=fock dtype=torch.float32 device=cuda
  flags: prefix_causal_registers=False, reverse_channel=True, n_registers=8

  Controls:
    [PASS] control_determinism: 0 logit (tol 0)
    [PASS] control_placebo: 0 logit (tol 0)
    [PASS] control_positive: 4.31 logit (tol 0.0001)

  Probes:
    [FAIL] future_perturbation: 0.286 logit (tol 0)
    [FAIL] target_relocation: 3.51 nats (tol 0.001)

    standard PPL 7.69  ->  honest PPL 258.07   (+3.510 nats, 33.4x inflation)

  Diagnostics (not part of the verdict):
    [PASS] mediation: 1 fraction (tol 0.9)
    leak attribution (mean effect 0.0231):
       100.0% removed by knocking out reverse_channel_scale   (residual 0)
         0.0% removed by knocking out creation_gate_qkv   (residual 0.0231)

  VERDICT: LEAK
```

## What it checks

Two probes, asking deliberately different questions:

**`future_perturbation`** — replace the future, then measure whether the past
moved. For a structurally causal model the answer is bit-exact `0.0`, so this
needs no tolerance tuning. It answers *is there a leak at all?*

**`target_relocation`** — score each target token twice: once with the whole
sequence supplied (standard perplexity) and once with only the prefix supplied,
so the future is physically absent (honest perplexity). The gap is the leak tax
in nats. It answers *how much perplexity was unearned?*

Both are needed. The register leak moved logits only slightly under far-future
topic swaps while smuggling in an enormous next-token advantage, so the first
probe alone understated it by orders of magnitude.

Once a leak is found, `mediation` asks *where* it lives. It re-runs the same
measurement with each candidate component clamped off — the controlled direct
effect — and reports the fraction of the leak each knockout removes. If one
component takes the residual to zero, that component is the locus.

Two details make the number trustworthy. Every arm reuses the *same*
counterfactual futures, so attribution cannot pick up RNG variation. And
attribution is computed on the mean effect rather than the max: `linf` is right
for detection because a causal model gives bit-exact zero there, but it is a max
over positions, so two partially-cancelling channels can make removing the
weaker one *raise* it and report a negative share.

Mediation is a **diagnostic** and never affects the verdict. Where a leak lives
is a different question from whether one exists, and good attribution must not
make a leaky model read as healthier.

## Monitoring a training run

A one-off audit certifies a checkpoint, not a run. Leak channels are typically
gated by a learnable scale initialised at zero, so the architecture leaks from
step one while the measured effect is nil, and the valve opens gradually as the
optimiser discovers the channel pays. By the time the final checkpoint is
audited, every intermediate number in the log is already contaminated and
nothing in the loss curve says when it happened.

This is not hypothetical for the Fock models: `reverse_channel_scale` is
initialised to exactly `0.0` behind a `tanh`, so at step zero the channel is
shut and no probe can see it.

```python
monitor = scaf.LeakMonitor(
    model, tokens=val_tokens, interval=20_000,
    seq_len=512, micro_batch=2, jsonl_path=report_path,
)
for step in range(total_steps):
    ...
    monitor.maybe_run(step)          # returns a JSONL-ready dict, or None
```

```
     step  verdict         linf        AILE   tau_leak
        0  CLEAN      0.000e+00   0.000e+00     0.0000
    20000  CLEAN      0.000e+00   0.000e+00     0.0000
    40000  LEAK       2.069e-02   3.007e-03     0.4113
  first leak detected at step 40000; every metric logged from that
  point on was measured through it
```

Three properties make it safe to leave on. Global RNG state is saved and
restored around every measurement, so a monitored run follows the *same*
trajectory as an unmonitored one — without that, enabling monitoring would
change which batches and dropout masks the model sees. Probe failures are
caught and recorded as an `ERROR` verdict rather than killing the job. And the
corpus is rewound before each run, so a change in AILE is a change in the model
rather than in the draw.

Scheduling is measured from the last run rather than by divisibility, so a run
resumed at step 7000 probes on schedule instead of waiting for the next
multiple.

## Formal estimands: DoWhy, EconML, refutation

The probe battery gives a verdict. `scaf.build_leak_frame` gives a **dataset**,
so the same intervention can be handed to a causal-inference engine and
interrogated with the standard verbs.

```python
frame = scaf.build_leak_frame(model, tokens=val_tokens, seq_len=512)
print(frame.summary())                        # torch only, no extra deps
print(scaf.estimate_leak(frame).summary())    # needs the [pywhy] extra
```

Each row is one scored target position under one arm — factual, or
`do(future := resampled)` — with the negative log-likelihood of the true next
token as the outcome, so the ATE is directly in nats and comparable to the
honest-perplexity gap.

**Identification is free, and the docs say so.** DoWhy's headline capability is
deciding whether an effect is estimable despite confounding. SCAF gets nothing
from it: the auditor assigns the arm, so no arrow points into the treatment and
the paired difference is already the causal effect. What the stack is actually
worth importing for is refutation, heterogeneity, and an independent estimator
that has no freedom to disagree with the paired difference — a mismatch means
the setup is wrong, and the report flags it.

```
SCAF estimand — do(future) on next-token NLL
  ATE            +14.6692 nats
  95% CI         [+9.60355, +20.2899]   (paired bootstrap)
  p-value        9.999e-05   (sign-flip, exact under pairing)
  paired ref.    +14.6692 nats   (agrees)

  Refutations:
    [PASS] placebo_treatment_refuter: new effect -1.078 (expected ~0)
    [PASS] random_common_cause: new effect +14.66 (expected ~14.67)

  Heterogeneity (exact stratified profile):
    tau by distance_to_cut:   [CausalForestDML in brackets]
      distance_to_cut=0            +107.6 nats  ( 36 pairs)   [+108.6]
      distance_to_cut=1                +0 nats  ( 12 pairs)   [+0.6062]
      distance_to_cut=2                +0 nats  ( 24 pairs)   [-0.1473]
      ...
      peak at distance_to_cut=0 (+107.6 nats)
```

`distance_to_cut` is the diagnostic axis. A next-token peek is a spike at zero
and nothing else; a shared-register leak spikes at the cut and decays; a
global-pool leak is roughly flat.

Three choices here are deliberate and worth knowing about:

- **Inference is paired, not pooled.** Leak effects are wildly
  heteroscedastic — one position can carry a hundred nats while every other
  carries exactly zero. DoWhy's own permutation test scrambles the treatment
  label across the whole frame, discards the pairing, and returned `p = 0.69`
  for an effect its own interval put at `[10.4, 18.5]`. That is a false
  all-clear. SCAF reports an exact sign-flip permutation test over pair
  differences instead, and keeps DoWhy's version behind `dowhy_inference=True`.
- **The outcome is centred within each pair.** Raw NLL varies by nats across
  positions for reasons having nothing to do with the treatment. Centring
  removes about 90% of that spread on a realistic model while leaving the ATE
  bit-identical.
- **The CATE default is a causal forest, not `LinearDML`.** Fitted to a spiky
  leak profile, the linear model reports a third of the true peak and a
  spurious *negative* effect far from the cut — the future appearing to
  suppress the past. The forest recovers the peak to within 1%. An exact
  model-free stratified profile is reported either way; the fit is a smoother
  over it, not a better measurement of it.

## Why the controls matter more than the probes

A leak probe reporting zero means either the model is causal, or **the probe
never looked**. Nothing in the number distinguishes them, and the second is
exactly what happened during the original audit.

So SCAF runs three controls and refuses to certify without them:

| Control | Guards against |
| --- | --- |
| `control_determinism` | a non-reproducible forward, which makes every threshold meaningless |
| `control_placebo` | a harness that fabricates differences — false alarms |
| `control_positive` | a probe not reaching the model — **false all-clears** |

The positive control changes a token the model *is* allowed to see and requires
a large response. If that fails, the verdict is `INVALID`, never `CLEAN`.

The verdict is tri-state on purpose. `INVALID` outranks `LEAK`: when the controls
fail, a clean model and a blind probe are indistinguishable, so claiming either
would be unearned. Missing evidence never reads as good news.

## Interfacing with models

SCAF requires no base class, no `Protocol`, and no changes to model code.
Adapters are resolved by **structural (duck) typing** — attribute and class-name
inspection, never `isinstance` — so SCAF installs and tests with no research repo
present. Resolution order: an explicit `adapter=` argument, then a model-supplied
`__scaf_adapter__()` hook, then registered adapters by descending priority,
ending at `GenericAdapter`.

The generic fallback needs only that `model(x)` accepts `(B, T)` longs and
returns logits `(B, T, V)`, directly or as element 0 of a tuple. That covers
every SemSimula model and any plain GPT-style decoder.

Full rationale, including the family inheritance tree and per-family intervention
points, is in [`docs/model-interface-design.md`](docs/model-interface-design.md).

## Custom adapters

```python
import scaf

@scaf.register_adapter
class MyAdapter(scaf.ModelAdapter):
    name, priority = "mine", 50

    @classmethod
    def detect(cls, model):
        return hasattr(model, "my_distinctive_attribute")

    def capabilities(self, model):
        return scaf.Capabilities(mediators=("my_gate",))
```

## Development

```bash
pip install -e ".[dev]"
pytest
ruff check .
python tools/lint_markdown.py README.md docs/*.md   # GitHub KaTeX/Mermaid rules
```

The suite runs without the `[pywhy]` extra; the DoWhy/EconML tests skip. To run
them, install `".[dev,pywhy]"`.

Integration tests against the real Fock-PARFLM models are marked `semsimula`
and skip unless the research repo is present. Point `SEMSIMULA_PAPER` at it, or
keep it as a sibling checkout. They assert the finding SCAF was built around:
with the reverse-channel gate open, the legacy register implementation reports
`LEAK` while `prefix_causal_registers=True` gives bit-exact zero.

## Status

Alpha, feature-complete for the planned scope. Implemented and tested: the
probe battery (future perturbation, target relocation, mediation), the
controls, the adapters, the scorecard, the training-loop monitor, and the
DoWhy/EconML estimand bridge with exact paired inference.

Not yet built: empirical DAG discovery via `causal-learn`, and adapters beyond
Fock and the generic fallback.

## License

MIT
