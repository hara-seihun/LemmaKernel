"""Runs a module's cases: the tests and the benchmark, for every module, without per-module scripts.

A module ships `cases.py` with

    def cases(ctx, rng) -> list[Case]      # inputs: families, arguments, which ops and reductions
    def invariants(ctx) -> None            # optional: cross-operation identities on larger inputs

and its manifest declares typed arguments, Lean value constructors, and `[[rejections]]`.
From those this file derives everything the old per-module test and bench scripts did:

- every case × every backend × every allowed reduction, stated as a Lean claim over the
  module's reference and checked by `decide +kernel` (`tools/leancheck.py`);
- the naive implementation on the same cases, against the same oracle;
- every `[[rejections]]` entry refused by the runtime with the declared message, and called
  `.invalid` by the reference where the request can be rendered;
- thread invariance and interchange roundtrips on every case;
- coverage: every (operation, reduction) pair the manifest allows must have an oracle case;
- the benchmark: kernel (one thread, every core) against naive on cases that carry `bench`,
  recorded in the module's `bench.json` together with a fingerprint of everything the numbers
  depend on (the module tree, the module trees it includes, the runtime), so `tools/bench.py`
  reruns only what changed.

`tests/test_cases.py` is the pytest entry; `tools/bench.py` the benchmark entry.
"""
from __future__ import annotations

import datetime
import hashlib
import importlib.util
import itertools
import json
import os
import platform
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))
import lemmakernel as lk  # noqa: E402
from lemmakernel import interchange as ic  # noqa: E402
from lemmakernel._manifest import MODULES, RUNTIME  # noqa: E402

RUNTIME_FAMILIES = {f["name"]: f for f in RUNTIME["families"]}
REDUCTIONS = {r["name"]: r for r in RUNTIME["reductions"]}
KIND_LEAN = ({k["name"]: k.get("lean") for k in RUNTIME.get("kinds", [])} |
             {k["name"]: k.get("lean") for m in MODULES for k in m.get("kinds", [])})


@dataclass
class Case:
    """One request shape. `op` may omit the module prefix. `reductions` defaults to every
    reduction the manifest allows for the op. `bench` names the reduction to benchmark, or None.
    `oracle` is False for inputs too large for the Lean kernel (they still get every other check).
    Several cases may share a name (the same inputs under different operations); the harness
    checks them in one Lean file. A large family may be given as a zero-argument callable so
    that tests which never touch it do not pay to build it."""
    name: str
    family: object
    op: str
    args: dict = field(default_factory=dict)
    reductions: list[str] | None = None
    what: str = ""
    bench: str | None = None
    oracle: bool = True

    @property
    def fam(self) -> lk.Handle:
        if callable(self.family):
            self.family = self.family()
        return self.family


@dataclass
class Module:
    manifest: dict
    dir: Path

    @property
    def name(self) -> str:
        return self.manifest["module"]["name"]

    @property
    def lean(self) -> str:
        n = self.name
        return n[:1].upper() + n[1:]

    @property
    def backends(self) -> list[str]:
        return [b for b in lk.describe()["available_backends"] if b.startswith(self.name + ".")]

    def operation(self, op: str) -> dict:
        bare = op.removeprefix(self.name + ".")
        for o in self.manifest["operations"]:
            if o["name"] == bare:
                return o
        raise KeyError(f"{self.name} has no operation {bare}")

    def allowed_reductions(self, op: str) -> list[str]:
        value = self.operation(op)["value"]
        return [r["name"] for r in RUNTIME["reductions"] if value in r["accepts"] or "*" in r["accepts"]]

    def _load(self, key: str, attr: str):
        path = self.dir / self.manifest["module"][key]
        spec = importlib.util.spec_from_file_location(f"{self.name}_{attr}", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def naive(self):
        return self._load("naive", "naive")

    def cases(self, ctx, rng) -> list[Case]:
        out = self._load("cases", "cases").cases(ctx, rng)
        for c in out:
            c.op = self.name + "." + c.op.removeprefix(self.name + ".")
        return out

    def invariants(self):
        return getattr(self._load("cases", "invariants"), "invariants", None)


def modules() -> list[Module]:
    return [Module(m, ROOT / "modules" / m["module"]["name"]) for m in MODULES]


def module(name: str) -> Module:
    return next(m for m in modules() if m.name == name)


# ---- fingerprints and bench records -------------------------------------------------------------

RUNTIME_INPUTS = ["runtime", "python/lemmakernel", "tools/harness.py", "tools/bench.py", "CMakeLists.txt",
                  "lean-toolchain", "lake-manifest.json"]
SKIP_DIRS = {"__pycache__", ".lake", ".pytest_cache"}
MODULE_REF = re.compile(r"(?:modules/|\"modules\" / \"|(?:\.\./)+)([a-z][a-z0-9_]*)[/\"]")


def _files(root: Path):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS)
        for f in sorted(filenames):
            if not f.endswith(".pyc"):
                yield Path(dirpath) / f


def _digest(h: hashlib._Hash, paths) -> None:
    for p in paths:
        for f in (_files(p) if p.is_dir() else [p]):
            if f.name == "bench.json" or not f.exists():
                continue
            h.update(str(f.relative_to(ROOT)).encode())
            h.update(b"\0")
            h.update(f.read_bytes())
            h.update(b"\0")


def module_dependencies(mod: Module) -> list[str]:
    """Other modules whose files this one's sources name (C++ includes, naive imports), transitively."""
    names = {m.name for m in modules()}
    seen, todo = set(), [mod.name]
    while todo:
        name = todo.pop()
        for f in _files(ROOT / "modules" / name):
            if f.suffix not in {".py", ".cpp", ".hpp", ".h", ".lean", ".toml", ".cmake", ".txt"}:
                continue
            for ref in MODULE_REF.findall(f.read_text(errors="replace")):
                if ref in names and ref != mod.name and ref not in seen:
                    seen.add(ref)
                    todo.append(ref)
    return sorted(seen)


def runtime_fingerprint() -> str:
    h = hashlib.sha256()
    _digest(h, [ROOT / p for p in RUNTIME_INPUTS])
    return h.hexdigest()


def module_fingerprint(mod: Module) -> dict:
    """`module` covers the module tree and the trees it depends on; `combined` adds the runtime."""
    h = hashlib.sha256()
    deps = module_dependencies(mod)
    _digest(h, [mod.dir] + [ROOT / "modules" / d for d in deps])
    module = h.hexdigest()
    runtime = runtime_fingerprint()
    return {"module": module, "runtime": runtime, "dependencies": deps,
            "combined": hashlib.sha256((module + runtime).encode()).hexdigest()}


def bench_record_path(mod: Module) -> Path:
    return mod.dir / "bench.json"


def bench_record(mod: Module) -> dict | None:
    p = bench_record_path(mod)
    return json.loads(p.read_text()) if p.exists() else None


def bench_staleness(mod: Module) -> str | None:
    """None when bench.json matches the current tree; otherwise why it does not."""
    rec = bench_record(mod)
    if rec is None:
        return "no bench.json"
    fp = module_fingerprint(mod)
    if rec["fingerprint"]["module"] != fp["module"]:
        return "module or dependency sources changed"
    if rec["fingerprint"]["runtime"] != fp["runtime"]:
        return "runtime changed"
    return None


def machine_description(threads: int) -> dict:
    cpu = ""
    try:
        cpu = next(l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name"))
    except (OSError, StopIteration):
        pass
    return {"host": platform.node(), "cpu": cpu, "threads": threads}


def git_commit() -> str:
    proc = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True)
    return proc.stdout.strip() if proc.returncode == 0 else ""


def write_bench_record(mod: Module, rows: list[dict], naive_limit: float, threads: int) -> Path:
    rec = {"module": mod.name, "fingerprint": module_fingerprint(mod),
           "measured_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
           "commit": git_commit(), "machine": machine_description(threads), "naive_limit_s": naive_limit,
           "rows": rows}
    p = bench_record_path(mod)
    p.write_text(json.dumps(rec, indent=1) + "\n")
    return p


# ---- Lean terms ---------------------------------------------------------------------------------

def L(x) -> str:
    """Lean literal for nested lists of ints, Options and tuples."""
    if x is None:
        return "none"
    if isinstance(x, tuple):
        return "(" + ", ".join(L(v) for v in x) + ")"
    if isinstance(x, list):
        return "[" + ", ".join(L(v) for v in x) + "]"
    if hasattr(x, "tolist"):
        return L(x.tolist())
    value = int(x)
    return f"({value})" if value < 0 else str(value)


def vectors_of(m: ic.Matrix | ic.Perms) -> list:
    """A batch of 1 x n rows, one k x n matrix, or a permutation batch as vectors."""
    if isinstance(m, ic.Perms):
        return m.tolist()
    return [r[0] for r in m.tolist()] if m.rows == 1 else m.member(0)


def encoded_matrices(m: ic.Matrix) -> list:
    """Lean's shared Mat type is Nat-valued, so signed matrices use their interchange ZigZag words."""
    if m.p != ic.GRAMS:
        return m.tolist()
    return [[[2 * x if x >= 0 else -2 * x - 1 for x in row] for row in a] for a in m.tolist()]


def lean_family(f: ic.Family) -> str:
    q, ctor = f.params, RUNTIME_FAMILIES[f.kind]["lean"]
    if f.kind == "explicit":
        (b,) = f.children
        if isinstance(b, ic.Perms):
            return f"(.{ctor} 0 {L([[g] for g in b.tolist()])})"
        return f"(.{ctor} {b.p} {L(encoded_matrices(b))})"
    if f.kind == "subsets":
        (d,) = f.children
        p = 0 if isinstance(d, ic.Perms) else d.p
        return f"(.{ctor} {p} {L(vectors_of(d))} {q['k']})"
    if f.kind == "grassmannian":
        return f"(.{ctor} {q['p']} {q['n']} {q['h']})"
    if f.kind == "all_matrices":
        return f"(.{ctor} {q['p']} {q['rows']} {q['cols']})"
    if f.kind == "transform":
        inner, c = f.children
        return f"(.{ctor} {lean_family(inner)} {L(c.member(0))})"
    if f.kind == "stack":
        inner, rows = f.children
        return f"(.{ctor} {lean_family(inner)} {L(rows.member(0))})"
    if f.kind == "group_elements":
        (gens,) = f.children
        if isinstance(gens, ic.Perms):
            return f"(.{ctor} 0 {L([[g] for g in gens.tolist()])})"
        return f"(.{ctor} {gens.p} {L(gens.tolist())})"
    if f.kind == "group_tables":
        (tables,) = f.children
        return f"(.{ctor} {L(tables.tolist())})"
    if f.kind == "subsets_of":
        (inner,) = f.children
        return f"(.{ctor} {lean_family(inner)} {q['k']})"
    if f.kind in ("symmetric_matrices", "alternating_matrices"):
        return f"(.{ctor} {q['p']} {q['n']})"
    if f.kind == "range":
        return f"(.{ctor} {q['a']} {q['b']})"
    if f.kind == "words":
        return f"(.{ctor} {q['alphabet']} {q['length']})"
    if f.kind == "latin_squares":
        return f"(.{ctor} {q['n']})"
    if f.kind == "partitions":
        return (f"(.{ctor} {q['total']} {q['max_part']} {q['max_parts']} {q['max_multiplicity']} "
                f"{q['distinct']} {q['odd']})")
    if f.kind == "compositions":
        return f"(.{ctor} {q['total']} {q['parts']} {q['max_part']})"
    if f.kind == "standard_tableaux":
        (shape,) = f.children
        return f"(.{ctor} {L(shape.member(0)[0])})"
    if f.kind == "all_graphs":
        return f"(.{ctor} {q['n']})"
    if f.kind == "edge_subgraphs":
        (host,) = f.children
        return f"(.{ctor} {L(host.member(0))} {q['k']})"
    if f.kind == "cayley_graphs":
        (gens,) = f.children
        return f"(.{ctor} {L(gens.tolist())})"
    if f.kind == "sublattices":
        (g,) = f.children
        return f"(.{ctor} {L(encoded_matrices(g)[0])} {q['index']})"
    raise ValueError(f.kind)


def lean_arg(typ: str, v) -> str:
    v = v.value() if isinstance(v, lk.Handle) else v
    if typ == "int":
        return str(int(v))
    if typ == "vector":
        return L(v.member(0)[0])
    if typ == "vectors":
        return L(vectors_of(v))
    if typ == "perms":
        return L(v.tolist())
    if typ == "group":
        return f"(.perms {L(v.tolist())})" if isinstance(v, ic.Perms) else f"(.mats {v.p} {L(v.tolist())})"
    if typ == "family":
        return lean_family(v)
    raise ValueError(typ)


def camel(name: str) -> str:
    head, *rest = name.split("_")
    return head + "".join(w[:1].upper() + w[1:] for w in rest)


def lean_call(ctor: str, decl_args: dict, args: dict) -> str:
    """`(.isPacking (n := 4) (h := 2))`. Arguments are passed by name: the binder of the Lean
    constructor is the camelCase of the manifest's argument name (`clique_size` is `cliqueSize`),
    and nothing else about the two declarations has to agree. Lean rejects the claim when a name
    is wrong, rather than an operation quietly reading its arguments in another order."""
    parts = [f".{ctor}"] + [f"({camel(a)} := {lean_arg(t, args[a])})" for a, t in decl_args.items()]
    return parts[0] if len(parts) == 1 else "(" + " ".join(parts) + ")"


def lean_op(mod: Module, op: str, args: dict) -> str:
    decl = mod.operation(op)
    return lean_call(camel(decl["name"]), decl.get("args", {}), args)


def lean_red(red: str, args: dict) -> str:
    return lean_call(red, REDUCTIONS[red].get("args", {}), args)


def lean_result(r) -> str:
    if isinstance(r, ic.Integers):
        return f".integers {L(r.values)}"
    if isinstance(r, ic.Count):
        assert r.visited == r.family_size, "incomplete enumeration reported"
        return f".count {r.value} {r.family_size}"
    if isinstance(r, ic.Histogram):
        assert r.visited == r.family_size, "incomplete enumeration reported"
        return f".histogram {r.family_size} {L(r.bins)}"
    if isinstance(r, ic.Hits):
        assert r.visited == r.family_size and r.total == len(r.indices)
        return f".hits {r.family_size} {L(r.indices)} {L(encoded_matrices(r.members))}"
    if isinstance(r, ic.First):
        if not r.found:
            assert r.visited == r.family_size, "incomplete enumeration reported"
            return f".first {r.family_size} none"
        assert r.visited == r.index + 1
        return f".first {r.family_size} (some ({r.index}, {L(encoded_matrices(r.member)[0])}))"
    if isinstance(r, ic.Extremum):
        assert r.visited == r.family_size, "incomplete enumeration reported"
        return f".extremum {r.family_size} {r.value} {r.index} {L(encoded_matrices(r.member)[0])}"
    ctor = KIND_LEAN.get(ic.kind_of(r))
    if ctor is None:
        raise TypeError(f"{ic.kind_of(r)} declares no Lean value constructor")
    return f".values [{', '.join(f'.{ctor} {L(r.member(i))}' for i in range(r.count))}]"


def claim(mod: Module, op: str, family: ic.Family, red: str, args: dict) -> str:
    return f"run {lean_op(mod, op, args)} {lean_family(family)} {lean_red(red, args)}"


def renderable(mod: Module, op: str, red: str, args: dict) -> bool:
    decl = mod.operation(op)
    needed = list(decl.get("args", {})) + list(REDUCTIONS[red].get("args", {}))
    return all(a in args for a in needed)


# ---- running cases ------------------------------------------------------------------------------

def request_args(mod: Module, op: str, red: str, args: dict) -> dict:
    """The case's arguments that this op and reduction declare; a case may carry more (e.g. a
    `limit` that only `hits` uses)."""
    wanted = set(mod.operation(op).get("args", {})) | set(REDUCTIONS[red].get("args", {}))
    return {k: v for k, v in args.items() if k in wanted}


def naive_args(args: dict) -> dict:
    return {k: (v.value() if isinstance(v, lk.Handle) else v) for k, v in args.items()}


def reductions_of(mod: Module, case: Case) -> list[str]:
    return case.reductions if case.reductions is not None else mod.allowed_reductions(case.op)


def safe_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_")


def coverage_gaps(mod: Module, cases: list[Case]) -> list[str]:
    """(op, reduction) pairs the manifest allows that no oracle case exercises."""
    covered = {(c.op, r) for c in cases if c.oracle for r in reductions_of(mod, c)}
    gaps = []
    for o in mod.manifest["operations"]:
        op = f"{mod.name}.{o['name']}"
        for r in mod.allowed_reductions(op):
            if (op, r) not in covered:
                gaps.append(f"{op}/{r}")
    return gaps


class Declined(Exception):
    """The context's backend does not accept the case's requests (a backend that serves part of
    its module, such as a GPU backend for one operation). Callers skip the case for that backend
    and check that the backend served something."""


def claims_for_case(mod: Module, ctx, case: Case, lc, naive=None) -> None:
    """Add the kernel's answers as claims. With `naive`, also run the naive implementation: when
    it agrees with the kernel byte for byte the one claim covers both; when it disagrees both
    answers are claimed and Lean says which (if either) is right. Raises Declined when the
    context's backend does not accept the request."""
    desc = case.fam.value()
    for red in reductions_of(mod, case):
        args = request_args(mod, case.op, red, case.args)
        try:
            h = ctx.run(case.op, case.fam, red, **args)
        except lk.Error as e:
            if e.status == 2 and "does not accept" in str(e):
                raise Declined(str(e)) from None
            raise
        lc.claim(claim(mod, case.op, desc, red, args), lean_result(h.value()), f"{case.name} {case.op}/{red}")
        if naive is not None:
            n = naive.run(case.op, desc, red, **naive_args(args))
            if n.encode() != h.export():
                lc.claim(claim(mod, case.op, desc, red, args), lean_result(n), f"naive {case.name} {case.op}/{red}")


def rotate(mod: Module, op: str, i: int) -> list[str]:
    """One of the op's allowed reductions, chosen by `i`, so a run of cases covers them all
    without every case paying for every reduction."""
    allowed = mod.allowed_reductions(op)
    return [allowed[i % len(allowed)]]


def rejection_requests(mod: Module, cases: list[Case]):
    """(rejection, case, op, reduction) for each manifest rejection."""
    for rej in mod.manifest.get("rejections", []):
        named = [c for c in cases if c.name == rej["case"]]
        if not named:
            raise KeyError(f"{mod.name}: rejection names unknown case {rej['case']!r}")
        if "op" in rej:
            op = mod.name + "." + rej["op"].removeprefix(mod.name + ".")
            case = next((c for c in named if c.op == op), named[0])
        else:
            if len(named) > 1:
                raise KeyError(f"{mod.name}: rejection for {rej['case']!r} must name `op`; the case covers {[c.op for c in named]}")
            case, op = named[0], named[0].op
        red = rej.get("reduction", (case.reductions or ["all"])[0])
        yield rej, case, op, red


def bench_case(mod: Module, ctx, case: Case, naive, naive_limit: float, threads: int) -> dict:
    """Time naive (full or extrapolated from a prefix) and the kernel at 1 and `threads` threads."""
    red = case.bench
    size = ctx.size(case.fam)
    desc = case.fam.value()
    nargs = naive_args(request_args(mod, case.op, red, case.args))
    prefix = min(size, 200)
    t = time.perf_counter()
    naive.run(case.op, desc, red, prefix=prefix, **nargs)
    per = (time.perf_counter() - t) / max(prefix, 1)
    if per * size <= naive_limit:
        t = time.perf_counter()
        naive_result = naive.run(case.op, desc, red, **nargs)
        naive_s, extrapolated = time.perf_counter() - t, False
    else:
        naive_result, naive_s, extrapolated = None, per * size, True
    times = {}
    exports = set()
    answer = None
    for n in (1, threads):
        ctx.threads = n
        best = None
        for _ in range(3):  # best of three: sub-millisecond runs are otherwise noise
            t = time.perf_counter()
            h = ctx.run(case.op, case.fam, red, **request_args(mod, case.op, red, case.args))
            best = min(best or 1e9, time.perf_counter() - t)
            exports.add(h.export())
        times[n] = best
        answer = h
    assert len(exports) == 1, f"{case.name}: thread count changed the answer"
    if naive_result is not None:
        assert naive_result.encode() == answer.export(), f"{case.name}: kernel disagrees with naive"
    return {"case": case.name, "what": case.what, "op": case.op, "reduction": red, "members": size,
            "naive_s": naive_s, "naive_extrapolated": extrapolated,
            "kernel_1_thread_s": times[1], "kernel_all_threads_s": times[threads], "threads": threads,
            "speedup_1_thread": naive_s / times[1], "speedup_all_threads": naive_s / times[threads],
            "answer": repr(answer.value())[:120]}


def format_bench_row(r: dict) -> str:
    return (f"{r['case']:32s} {r['members']:>12,d} members  naive {r['naive_s']:9.2f}s{'~' if r['naive_extrapolated'] else ' '}  "
            f"kernel x1 {r['kernel_1_thread_s']:8.3f}s  x{r['threads']} {r['kernel_all_threads_s']:8.3f}s  "
            f"speedup {r['speedup_1_thread']:9.0f}x / {r['speedup_all_threads']:9.0f}x\n    {r['what']}\n    {r['answer']}")


# ---- input builders shared by cases.py files ----------------------------------------------------

def random_batch(rng, p, count, rows, cols) -> ic.Matrix:
    """Batches with a mix of generic, singular and structured members."""
    mats = []
    for i in range(count):
        m = [[rng.randrange(p) for _ in range(cols)] for _ in range(rows)]
        if i % 4 == 1 and rows > 1:
            m[-1] = list(m[0])
        if i % 4 == 2:
            m[0] = [0] * cols
            for r in m:
                r[-1] = 0
        if i % 4 == 3:
            m = [[rng.randrange(p) if rng.random() < 0.3 else 0 for _ in range(cols)] for _ in range(rows)]
        mats.append(m)
    return lk.matrix(p, mats)


def unit_vectors(p, n) -> ic.Matrix:
    return lk.matrix(p, [[int(i == j) for j in range(n)] for i in range(n)])


def cyclic(n):
    return [[(i + 1) % n for i in range(n)]]


def dihedral(n):
    return [[(i + 1) % n for i in range(n)], [(-i) % n for i in range(n)]]


def symmetric(n):
    return [[(i + 1) % n for i in range(n)], [1, 0] + list(range(2, n))]


def companion(coeffs, p):
    """Companion matrix of the monic polynomial x^n + c_{n-1} x^{n-1} + ... + c_0, acting on rows:
    multiplication by x in the basis 1, x, ..., x^{n-1} of F_p[x]/(f)."""
    n = len(coeffs)
    m = [[0] * n for _ in range(n)]
    for i in range(n - 1):
        m[i][i + 1] = 1
    m[n - 1] = [(-c) % p for c in coeffs]
    return m


def frobenius(coeffs, p):
    """v -> v^p on F_p[x]/(f) in the same basis; with `companion` it generates the normaliser of
    the Singer cycle, of order n (p^n - 1) when f is primitive."""
    n = len(coeffs)
    x = companion(coeffs, p)
    rows, power = [], [1] + [0] * (n - 1)
    for _ in range(n):
        rows.append(power)
        for _ in range(p):
            power = [sum(power[i] * x[i][j] for i in range(n)) % p for j in range(n)]
    return rows


def projective_points(ctx, p, n) -> ic.Matrix:
    """The points of PG(n-1, p) as normalised rows, in Grassmannian order."""
    return lk.matrix(p, [m[0] for m in ctx.value("gfp.rref", ctx.grassmannian(p, n, 1)).tolist()])
