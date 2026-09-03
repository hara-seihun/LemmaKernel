"""Interchange objects as Lean terms, shared by every module's tests.

The reference definitions live in each module's Reference.lean; these helpers only spell
Python-side objects (families, matrices, permutations) in the syntax those definitions expect.
"""
from __future__ import annotations

import lemmakernel as lk
from lemmakernel import interchange as ic


def L(x) -> str:
    """Lean literal for nested lists of ints / Options / tuples."""
    if x is None:
        return "none"
    if isinstance(x, tuple):
        return "(" + ", ".join(L(v) for v in x) + ")"
    if isinstance(x, (list,)):
        return "[" + ", ".join(L(v) for v in x) + "]"
    if hasattr(x, "tolist"):
        return L(x.tolist())
    return str(int(x))


def lean_family(f: ic.Family) -> str:
    q = f.params
    if f.kind == "explicit":
        (b,) = f.children
        return f"(.explicit {b.p} {L(b.tolist())})"
    if f.kind == "subsets":
        (d,) = f.children
        vecs = [m[0] for m in d.tolist()] if d.rows == 1 else d.member(0)
        return f"(.subsets {d.p} {L(vecs)} {q['k']})"
    if f.kind == "grassmannian":
        return f"(.grassmannian {q['p']} {q['n']} {q['h']})"
    if f.kind == "all_matrices":
        return f"(.allMatrices {q['p']} {q['rows']} {q['cols']})"
    if f.kind == "transform":
        inner, c = f.children
        return f"(.transform {lean_family(inner)} {L(c.member(0))})"
    if f.kind == "stack":
        inner, rows = f.children
        return f"(.stack {lean_family(inner)} {L(rows.member(0))})"
    raise ValueError(f.kind)


def lean_red(red: str, args: dict) -> str:
    return f"(.hits {args['limit']})" if red == "hits" else "." + red



def lean_perms(perms: ic.Perms) -> str:
    return L(perms.tolist())


def random_batch(rng, p, count, rows, cols):
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


