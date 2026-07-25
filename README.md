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

## Status

Alpha. The probe battery, controls, adapters, and scorecard are implemented and
tested. Mediation attribution, the DoWhy/EconML estimand bridge, and the
training-loop monitor are planned.

## License

MIT
