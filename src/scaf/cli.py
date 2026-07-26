"""``scaf`` command line interface.

Three subcommands:

``scaf info``
    Report the installed version and which adapters are registered. Useful for
    confirming a Colab install picked up the right revision.

``scaf audit``
    Audit a checkpoint loaded through a user-supplied builder function. SCAF
    cannot import SemSimula model classes itself (the research repo is not a
    package), so the builder is given as ``module:function`` and is expected to
    return a ready model. This keeps the CI gate usable without SCAF taking a
    dependency on any particular research tree.

``scaf estimate``
    Build an interventional frame from the same model and report the effect as
    a formal estimand — exact paired inference always, plus DoWhy refutation
    and EconML heterogeneity when the ``[pywhy]`` extra is installed.
"""

from __future__ import annotations

import argparse
import importlib
import sys


def _load_tokens(path: str):
    """Load a ``.npy`` token file. NumPy is imported lazily — it is not a core
    dependency, and only this one code path needs it."""
    try:
        import numpy as np
    except ImportError:  # pragma: no cover - trivial guard
        raise SystemExit(
            "--tokens requires NumPy: pip install numpy"
        ) from None
    return np.load(path)


def _load_builder(spec: str):
    if ":" not in spec:
        raise SystemExit(
            f"--builder must be 'module:function', got {spec!r}"
        )
    mod_name, fn_name = spec.split(":", 1)
    mod = importlib.import_module(mod_name)
    fn = getattr(mod, fn_name, None)
    if fn is None:
        raise SystemExit(f"{mod_name!r} has no attribute {fn_name!r}")
    return fn


def _cmd_info(_args) -> int:
    import scaf
    from scaf.core.adapters import registered_adapters

    print(f"scaf {scaf.__version__}")
    print("registered adapters (highest priority first):")
    for cls in registered_adapters():
        print(f"  {cls.priority:>4}  {cls.name}")
    return 0


def _cmd_audit(args) -> int:
    import scaf

    model = _load_builder(args.builder)()
    tokens = _load_tokens(args.tokens) if args.tokens else None

    report = scaf.audit(
        model,
        tokens=tokens,
        device=args.device,
        dtype=args.dtype,
        seq_len=args.seq_len,
        n_seqs=args.n_seqs,
        n_targets=args.n_targets,
        micro_batch=args.micro_batch,
        seed=args.seed,
    )
    print(report.summary() if not args.json else report.to_json())

    if args.fail_on_leak and not report.passed:
        return 1
    return 0


def _cmd_estimate(args) -> int:
    import scaf

    frame = scaf.build_leak_frame(
        _load_builder(args.builder)(),
        tokens=_load_tokens(args.tokens) if args.tokens else None,
        device=args.device,
        dtype=args.dtype,
        seq_len=args.seq_len,
        n_seqs=args.n_seqs,
        n_pairs=args.n_pairs,
        max_positions=args.max_positions,
        micro_batch=args.micro_batch,
        seed=args.seed,
    )
    if args.csv:
        frame.to_csv(args.csv)
        print(f"wrote {len(frame)} rows to {args.csv}")
    print(frame.summary())

    if args.frame_only:
        return 0
    try:
        report = scaf.estimate_leak(frame, cate_model=args.cate_model or None)
    except ImportError as exc:
        # The frame and its exact inference are already printed and are the
        # substantive result, so a missing optional extra is a note rather
        # than a failure.
        print(f"\nskipping the PyWhy stage: {exc}")
        return 0
    print()
    print(report.summary())
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="scaf", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("info", help="show version and registered adapters") \
        .set_defaults(func=_cmd_info)

    a = sub.add_parser("audit", help="audit a model for causal leaks")
    a.add_argument("--builder", required=True,
                   help="'module:function' returning a ready nn.Module")
    a.add_argument("--tokens", help=".npy file of token ids")
    a.add_argument("--device", default="cpu")
    a.add_argument("--dtype", default=None,
                   help="float64 for architectural probes, float32 for checkpoints")
    a.add_argument("--seq-len", type=int, default=128, dest="seq_len")
    a.add_argument("--n-seqs", type=int, default=8, dest="n_seqs")
    a.add_argument("--n-targets", type=int, default=32, dest="n_targets")
    a.add_argument("--micro-batch", type=int, default=4, dest="micro_batch")
    a.add_argument("--seed", type=int, default=0)
    a.add_argument("--json", action="store_true", help="emit JSON, not a table")
    a.add_argument("--fail-on-leak", action="store_true",
                   help="exit 1 unless the verdict is CLEAN (CI gate)")
    a.set_defaults(func=_cmd_audit)

    e = sub.add_parser(
        "estimate", help="report the leak as a formal causal estimand"
    )
    e.add_argument("--builder", required=True,
                   help="'module:function' returning a ready nn.Module")
    e.add_argument("--tokens", help=".npy file of token ids")
    e.add_argument("--device", default="cpu")
    e.add_argument("--dtype", default=None)
    e.add_argument("--seq-len", type=int, default=128, dest="seq_len")
    e.add_argument("--n-seqs", type=int, default=8, dest="n_seqs")
    e.add_argument("--n-pairs", type=int, default=2, dest="n_pairs")
    e.add_argument("--max-positions", type=int, default=16,
                   dest="max_positions",
                   help="target positions scored per split; the row-count knob")
    e.add_argument("--micro-batch", type=int, default=4, dest="micro_batch")
    e.add_argument("--seed", type=int, default=0)
    e.add_argument("--csv", help="also write the tidy frame here")
    e.add_argument("--frame-only", action="store_true",
                   help="skip DoWhy/EconML; exact paired inference only")
    e.add_argument("--cate-model", default="forest",
                   choices=["forest", "linear", ""], dest="cate_model",
                   help="EconML estimator, or '' for the exact profile only")
    e.set_defaults(func=_cmd_estimate)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
