"""Every module's cases against every backend, the naive implementation, and the Lean reference.

No expected answers live anywhere in the repository. For each case the kernel's answer is stated
as a Lean `example` over the module's `run` and Lean's `decide +kernel` accepts or rejects it.
See tools/harness.py for what a case is and how the manifest drives the rest.

Run from the repository root after building: `pytest -n auto tests`.
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "python"))
import lemmakernel as lk  # noqa: E402
from tools import harness as H  # noqa: E402
from tools.leancheck import LeanCheck  # noqa: E402

MODULES = H.modules()
SEED = 1


def _case_names(mod: H.Module, oracle_only: bool) -> list[str]:
    cs = mod.cases(lk.Context(), random.Random(SEED))
    return sorted({c.name for c in cs if c.oracle or not oracle_only})


def _cases_named(mod: H.Module, ctx, name: str) -> list[H.Case]:
    return [c for c in mod.cases(ctx, random.Random(SEED)) if c.name == name]


def _params(oracle_only: bool, per_backend: bool):
    out = []
    for mod in MODULES:
        backends = mod.backends if per_backend else [None]
        for name in _case_names(mod, oracle_only):
            for b in backends:
                out.append(pytest.param(mod.name, b, name, id=f"{b or mod.name}:{name}"))
    return out


@pytest.mark.parametrize("module,backend,name", _params(oracle_only=True, per_backend=True))
def test_backend_and_naive_match_reference(module, backend, name):
    """Kernel answers as Lean claims; the naive implementation on the same requests, which only
    costs a second Lean claim where it disagrees with the kernel."""
    mod = H.module(module)
    ctx = lk.Context(backend)
    naive = mod.naive() if backend == mod.backends[0] else None
    lc = LeanCheck(f"{backend}_{H.safe_name(name)}", [f"{mod.lean}.Reference"], [mod.lean, "Lk"])
    served = 0
    for case in _cases_named(mod, ctx, name):
        try:
            H.claims_for_case(mod, ctx, case, lc, naive)
            served += 1
        except H.Declined as e:
            if backend == mod.backends[0]:
                raise AssertionError(f"the module's first backend must serve every case: {e}")
    if not served:
        pytest.skip(f"{backend} declines {name!r}")
    lc.verify()


@pytest.mark.parametrize("module", [m.name for m in MODULES])
def test_every_operation_and_reduction_has_an_oracle_case(module):
    mod = H.module(module)
    gaps = H.coverage_gaps(mod, mod.cases(lk.Context(), random.Random(SEED)))
    assert not gaps, f"{module}: no oracle case exercises {gaps}"


@pytest.mark.parametrize("module", [m.name for m in MODULES])
def test_rejections(module):
    """What the manifest says the runtime refuses, it refuses with that message, and the
    reference calls the same request invalid."""
    mod = H.module(module)
    ctx = lk.Context()
    cases = mod.cases(ctx, random.Random(SEED))
    lc = LeanCheck(f"rejections_{mod.name}", [f"{mod.lean}.Reference"], [mod.lean, "Lk"])
    for rej, case, op, red in H.rejection_requests(mod, cases):
        args = H.request_args(mod, op, red, case.args)
        with pytest.raises(lk.Error, match=rej["error"]):
            ctx.run(op, case.fam, red, **args)
        if H.renderable(mod, op, red, args):
            lc.claim(H.claim(mod, op, case.fam.value(), red, args), ".invalid", f"{case.name} {op}/{red} rejected")
    lc.verify()


@pytest.mark.parametrize("module", [m.name for m in MODULES])
def test_threads_and_roundtrips(module):
    """Thread count never changes an answer; every family and result survives export/load."""
    mod = H.module(module)
    ctx = lk.Context()
    rejected = {r["case"] for r in mod.manifest.get("rejections", [])}
    for case in mod.cases(ctx, random.Random(SEED)):
        if case.name in rejected or (not case.oracle and case.bench is None):
            continue
        reds = [case.bench] if case.bench else H.reductions_of(mod, case)
        if not case.oracle and ctx.size(case.fam) > 2_000_000:
            continue
        again = ctx.load(case.fam.export())
        assert again.export() == case.fam.export()
        for red in reds:
            args = H.request_args(mod, case.op, red, case.args)
            exports = set()
            for threads in (1, 3, 32):
                ctx.threads = threads
                h = ctx.run(case.op, case.fam, red, **args)
                exports.add(h.export())
            assert len(exports) == 1, f"{case.name} {case.op}/{red}: thread count changed the answer"
            assert ctx.load(h.export()).export() == h.export()


@pytest.mark.parametrize("module", [m.name for m in MODULES])
def test_invariants(module):
    mod = H.module(module)
    inv = mod.invariants()
    if inv is None:
        pytest.skip(f"{module} declares no invariants")
    inv(lk.Context())
