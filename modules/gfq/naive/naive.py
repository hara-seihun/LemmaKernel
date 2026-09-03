"""Readable polynomial-basis linear algebra over explicitly presented GF(q)."""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "python"))
from lemmakernel import naive as rt  # noqa: E402
from lemmakernel.interchange import Basis, Family, Inverses, Matrix, Solutions  # noqa: E402

SUPPORTED = {"explicit", "subsets", "subsets_of", "grassmannian", "all_matrices", "symmetric_matrices", "stack"}


def _smallest_divisor(q: int) -> int:
    for p in range(2, int(q**0.5) + 1):
        if q % p == 0:
            return p
    return q


def _digits(a: int, p: int, d: int) -> list[int]:
    out = []
    for _ in range(d):
        out.append(a % p)
        a //= p
    return out


def _encode(cs, p: int) -> int:
    out = 0
    for c in reversed(cs):
        out = out * p + c
    return out


def _remainder(f: list[int], g: list[int], p: int) -> list[int]:
    out = list(f)
    d = len(g) - 1
    for k in range(len(out) - 1, d - 1, -1):
        lead = out[k]
        for j in range(d):
            out[k - d + j] = (out[k - d + j] - lead * g[j]) % p
        out[k] = 0
    return out[:d]


def _irreducible(modulus: list[int], p: int) -> bool:
    d = len(modulus) - 1
    for k in range(1, d // 2 + 1):
        for code in range(p**k):
            if not any(_remainder(modulus, _digits(code, p, k) + [1], p)):
                return False
    return True


class Field:
    def __init__(self, q: int, modulus: Matrix):
        if modulus.p != q or modulus.count != 1 or modulus.rows != 1:
            raise ValueError("modulus must be one vector tagged with the same q as the family")
        self.q = q
        self.f = modulus.member(0)[0]
        self.p = _smallest_divisor(q)
        self.d = len(self.f) - 1
        if self.d < 1 or self.p**self.d != q:
            raise ValueError("q must equal p^d for the modulus degree d")
        if self.f[-1] != 1 or any(c >= self.p for c in self.f):
            raise ValueError("modulus must be monic with coefficients in F_p")
        if not _irreducible(self.f, self.p):
            raise ValueError("modulus must be irreducible over F_p")

    def add(self, a: int, b: int) -> int:
        return _encode([(x + y) % self.p for x, y in zip(_digits(a, self.p, self.d), _digits(b, self.p, self.d))], self.p)

    def neg(self, a: int) -> int:
        return _encode([(-x) % self.p for x in _digits(a, self.p, self.d)], self.p)

    def mul(self, a: int, b: int) -> int:
        aa, bb = _digits(a, self.p, self.d), _digits(b, self.p, self.d)
        out = [0] * (2 * self.d - 1)
        for i, x in enumerate(aa):
            for j, y in enumerate(bb):
                out[i + j] = (out[i + j] + x * y) % self.p
        return _encode(_remainder(out, self.f, self.p), self.p)

    def inv(self, a: int) -> int:
        out, base, e = 1, a, self.q - 2
        while e:
            if e & 1:
                out = self.mul(out, base)
            base = self.mul(base, base)
            e >>= 1
        return out

    def sub_mul(self, row, other, c):
        nc = self.neg(c)
        return [self.add(a, self.mul(nc, b)) for a, b in zip(row, other)]


def rref(rows: list[list[int]], field: Field):
    R = [list(row) for row in rows]
    pivots = []
    r = 0
    n = len(R[0]) if R else 0
    for c in range(n):
        pivot = next((i for i in range(r, len(R)) if R[i][c]), None)
        if pivot is None:
            continue
        R[r], R[pivot] = R[pivot], R[r]
        scale = field.inv(R[r][c])
        R[r] = [field.mul(x, scale) for x in R[r]]
        for i in range(len(R)):
            if i != r and R[i][c]:
                R[i] = field.sub_mul(R[i], R[r], R[i][c])
        pivots.append(c)
        r += 1
        if r == len(R):
            break
    return R, pivots


def rank(rows, field: Field) -> int:
    return len(rref(rows, field)[1])


def nullspace(rows, field: Field):
    R, pivots = rref(rows, field)
    n = len(rows[0])
    out = []
    for free in range(n):
        if free in pivots:
            continue
        v = [0] * n
        v[free] = 1
        for i, pivot in enumerate(pivots):
            v[pivot] = field.neg(R[i][free])
        out.append(v)
    return out


def solve(rows, rhs, field: Field):
    n = len(rows[0])
    R, pivots = rref([row + [b] for row, b in zip(rows, rhs)], field)
    if n in pivots:
        return None
    x = [0] * n
    for i, pivot in enumerate(pivots):
        x[pivot] = R[i][n]
    return x


def inverse(rows, field: Field):
    n = len(rows)
    augmented = [row + [int(i == j) for j in range(n)] for i, row in enumerate(rows)]
    R, pivots = rref(augmented, field)
    return [row[n:] for row in R] if pivots == list(range(n)) else None


def _flat(mats):
    return [x for matrix in mats for row in matrix for x in row]


def run(op: str, family: Family, reduction: str = "all", prefix: int | None = None, **args):
    op = op.removeprefix("gfq.")
    if family.kind not in SUPPORTED:
        raise ValueError(f"{op} is not defined on {family.kind} families")
    q = rt.prime(family)
    field = Field(q, args["modulus"])
    members = list(itertools.islice(rt.iter_members(family), prefix))
    count = len(members)
    rows = len(members[0]) if members else family.params.get("rows", 0)
    cols = len(members[0][0]) if members else family.params.get("cols", 0)

    if op == "rank":
        return rt.reduce_int(reduction, [rank(m, field) for m in members], members, q)
    if op == "in_span":
        target = args["target"].member(0)[0]
        if len(target) != cols:
            raise ValueError("target must have one row and member columns entries")
        flags = [rank(m, field) == rank(m + [target], field) for m in members]
        return rt.reduce_bool(reduction, flags, members, q, **args)
    if op == "rref":
        return Matrix(q, count, rows, cols, _flat([rref(m, field)[0] for m in members]))
    if op == "nullspace":
        bases = [nullspace(m, field) for m in members]
        offsets = [0]
        for basis in bases:
            offsets.append(offsets[-1] + len(basis))
        return Basis(q, count, cols, offsets, _flat(bases))
    if op == "solve":
        if family.kind != "explicit":
            raise ValueError("solve is defined on explicit families only")
        rhs: Matrix = args["rhs"]
        if rhs.p != q or rhs.count != count or rhs.rows != 1 or rhs.cols != rows:
            raise ValueError("rhs must be count vectors of member rows entries over the same field")
        solutions = [solve(m, rhs.member(i)[0], field) for i, m in enumerate(members)]
        return Solutions(q, count, cols, [int(x is not None) for x in solutions],
                         [a for x in solutions for a in (x or [0] * cols)])
    if op == "inverse":
        if family.kind != "explicit":
            raise ValueError("inverse is defined on explicit families only")
        if rows != cols:
            raise ValueError("inverse needs square members")
        inverses = [inverse(m, field) for m in members]
        zero = [[0] * rows for _ in range(rows)]
        return Inverses(q, count, rows, [int(x is not None) for x in inverses],
                        _flat([x or zero for x in inverses]))
    raise ValueError(f"unknown operation {op}")
