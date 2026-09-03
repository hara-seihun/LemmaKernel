"""The interchange encoding, in Python.

A blob is: magic "LKIF", u32 version (1), string kind, u32 parameter count, that many
(string name, u64 value) pairs, u64 payload length, payload. Strings are u32 length + UTF-8.
Everything is little-endian. Field labels are packed at the width `entry_width(p)`, where the
header parameter `p` carries the field size.

This module is used by the binding (to build inputs and read results), by the naive
implementation (so it consumes exactly what the kernel consumes), and by the tests (to compare
kernel output against naive output byte for byte).
"""
from __future__ import annotations

import math
import struct
from dataclasses import dataclass, field

MAGIC = b"LKIF"
VERSION = 1
NATURALS = (1 << 64) - 1  # Matrix.p for natural-number matrices (kind lk.naturals)

try:
    import numpy as _np
except ImportError:  # numpy is optional; entries fall back to lists
    _np = None


def entry_width(p: int) -> int:
    if p in (0, NATURALS):  # permutations: point indices; naturals
        return 4
    if p < 1 << 8:
        return 1
    if p < 1 << 16:
        return 2
    if p < 1 << 32:
        return 4
    return 8


def pack_entries(entries, p: int) -> bytes:
    w = entry_width(p)
    if _np is not None:
        arr = _np.ascontiguousarray(_np.asarray(entries, dtype=_np.uint64).ravel())
        return arr.astype({1: "<u1", 2: "<u2", 4: "<u4", 8: "<u8"}[w]).tobytes()
    fmt = {1: "B", 2: "<H", 4: "<I", 8: "<Q"}[w]
    return b"".join(struct.pack(fmt, int(e)) for e in entries)


def unpack_entries(buf: bytes, p: int, n: int):
    w = entry_width(p)
    if len(buf) != n * w:
        raise ValueError(f"expected {n * w} bytes of entries, got {len(buf)}")
    if _np is not None:
        return _np.frombuffer(buf, dtype={1: "<u1", 2: "<u2", 4: "<u4", 8: "<u8"}[w]).astype(_np.int64)
    fmt = {1: "B", 2: "H", 4: "I", 8: "Q"}[w]
    return list(struct.unpack(f"<{n}{fmt}", buf))


def _str(s: str) -> bytes:
    b = s.encode()
    return struct.pack("<I", len(b)) + b


def encode(kind: str, params: dict[str, int], payload: bytes) -> bytes:
    out = [MAGIC, struct.pack("<I", VERSION), _str(kind), struct.pack("<I", len(params))]
    for k in sorted(params):
        out.append(_str(k))
        out.append(struct.pack("<Q", int(params[k])))
    out.append(struct.pack("<Q", len(payload)))
    out.append(payload)
    return b"".join(out)


@dataclass
class Blob:
    kind: str
    params: dict[str, int]
    payload: bytes


def split(buf: bytes, offset: int = 0) -> tuple[Blob, int]:
    """Parse one blob starting at `offset`; return it and the offset after it."""
    if buf[offset:offset + 4] != MAGIC:
        raise ValueError("not an interchange blob")
    pos = offset + 4
    (version,) = struct.unpack_from("<I", buf, pos)
    pos += 4
    if version != VERSION:
        raise ValueError(f"unsupported interchange version {version}")
    (n,) = struct.unpack_from("<I", buf, pos)
    pos += 4
    kind = buf[pos:pos + n].decode()
    pos += n
    (nparams,) = struct.unpack_from("<I", buf, pos)
    pos += 4
    params = {}
    for _ in range(nparams):
        (n,) = struct.unpack_from("<I", buf, pos)
        pos += 4
        name = buf[pos:pos + n].decode()
        pos += n
        (params[name],) = struct.unpack_from("<Q", buf, pos)
        pos += 8
    (plen,) = struct.unpack_from("<Q", buf, pos)
    pos += 8
    return Blob(kind, params, buf[pos:pos + plen]), pos + plen


# ---- object kinds -------------------------------------------------------------------------------

@dataclass
class Matrix:
    """A batch of labelled field matrices. `p` is the field size; `entries` is flat, row-major."""
    p: int
    count: int
    rows: int
    cols: int
    entries: object  # numpy int64 array or list of ints, length count*rows*cols

    def member(self, i: int):
        n = self.rows * self.cols
        flat = self.entries[i * n:(i + 1) * n]
        return [[int(flat[r * self.cols + c]) for c in range(self.cols)] for r in range(self.rows)]

    def tolist(self):
        return [self.member(i) for i in range(self.count)]

    def encode(self) -> bytes:
        if self.p == NATURALS:
            return encode("lk.naturals", {"count": self.count, "rows": self.rows, "cols": self.cols},
                          pack_entries(self.entries, self.p))
        return encode("gfp.matrix", {"p": self.p, "count": self.count, "rows": self.rows, "cols": self.cols},
                      pack_entries(self.entries, self.p))


def naturals(data, rows: int | None = None, cols: int | None = None) -> Matrix:
    """A natural-number matrix batch (kind lk.naturals, entries < 2^32): the members of `range`
    and `words` families. Same shapes as `matrix`."""
    return matrix(NATURALS, data, rows, cols)


def matrix(p: int, data, rows: int | None = None, cols: int | None = None) -> Matrix:
    """Build a Matrix from nested lists: one matrix (rows x cols) or a list of them."""
    if _np is not None and isinstance(data, _np.ndarray):
        if data.ndim == 2:
            data = data[None]
        count, rows, cols = data.shape
        return Matrix(p, count, rows, cols, data.reshape(-1).astype(_np.int64) % p)
    if rows is not None and cols is not None:
        flat = [int(x) % p for x in data]
        if len(flat) % (rows * cols):
            raise ValueError("flat data length is not a multiple of rows*cols")
        return Matrix(p, len(flat) // (rows * cols), rows, cols, flat)
    data = list(data)
    if not data:
        raise ValueError("cannot infer shape from empty data")
    if data and isinstance(data[0], (list, tuple)) and data[0] and isinstance(data[0][0], (list, tuple)):
        mats = data
    else:
        mats = [data]
    rows, cols = len(mats[0]), len(mats[0][0])
    flat = []
    for m in mats:
        if len(m) != rows or any(len(r) != cols for r in m):
            raise ValueError("ragged batch")
        for r in m:
            flat.extend(int(x) % p for x in r)
    return Matrix(p, len(mats), rows, cols, flat)


@dataclass
class Perms:
    """A batch of `count` permutations of {0..n-1}; entry g*n+i is the image of point i under g."""
    n: int
    count: int
    entries: object

    def member(self, i: int):
        return [int(x) for x in self.entries[i * self.n:(i + 1) * self.n]]

    def tolist(self):
        return [self.member(i) for i in range(self.count)]

    def encode(self) -> bytes:
        return encode("orbits.perms", {"n": self.n, "count": self.count}, pack_entries(self.entries, 0))


def perms(n: int, data) -> Perms:
    """Build a Perms batch from a list of permutations (each a list of n images)."""
    data = [list(map(int, g)) for g in data]
    for g in data:
        if sorted(g) != list(range(n)):
            raise ValueError(f"{g} is not a permutation of 0..{n - 1}")
    return Perms(n, len(data), [x for g in data for x in g])


@dataclass
class GraphGroups:
    """One complete, lexicographically sorted automorphism group per graph."""
    count: int
    n: int
    offsets: list[int]
    entries: object

    def member(self, i: int):
        a, b = self.offsets[i], self.offsets[i + 1]
        return [[int(x) for x in self.entries[g * self.n:(g + 1) * self.n]] for g in range(a, b)]

    def encode(self) -> bytes:
        payload = struct.pack(f"<{len(self.offsets)}Q", *self.offsets) + pack_entries(self.entries, 0)
        return encode("graph_iso.groups", {"count": self.count, "n": self.n}, payload)


@dataclass
class Basis:
    p: int
    count: int
    cols: int
    offsets: list[int]
    entries: object

    def member(self, i: int):
        a, b = self.offsets[i], self.offsets[i + 1]
        return [[int(self.entries[(a + r) * self.cols + c]) for c in range(self.cols)] for r in range(b - a)]

    def encode(self) -> bytes:
        payload = struct.pack(f"<{len(self.offsets)}Q", *self.offsets) + pack_entries(self.entries, self.p)
        return encode("gfp.basis", {"p": self.p, "count": self.count, "cols": self.cols}, payload)


@dataclass
class Solutions:
    p: int
    count: int
    length: int
    solvable: list[int]
    entries: object

    def member(self, i: int):
        if not self.solvable[i]:
            return None
        return [int(x) for x in self.entries[i * self.length:(i + 1) * self.length]]

    def encode(self) -> bytes:
        return encode("gfp.solutions", {"p": self.p, "count": self.count, "length": self.length},
                      bytes(self.solvable) + pack_entries(self.entries, self.p))


@dataclass
class Inverses:
    p: int
    count: int
    n: int
    invertible: list[int]
    entries: object

    def member(self, i: int):
        if not self.invertible[i]:
            return None
        n = self.n
        flat = self.entries[i * n * n:(i + 1) * n * n]
        return [[int(flat[r * n + c]) for c in range(n)] for r in range(n)]

    def encode(self) -> bytes:
        return encode("gfp.inverses", {"p": self.p, "count": self.count, "n": self.n},
                      bytes(self.invertible) + pack_entries(self.entries, self.p))


@dataclass
class Witness:
    p: int
    count: int
    rows: int
    cols: int
    r: object
    t: object

    def member(self, i: int):
        rc, rr = self.rows * self.cols, self.rows * self.rows
        R = self.r[i * rc:(i + 1) * rc]
        T = self.t[i * rr:(i + 1) * rr]
        return ([[int(R[a * self.cols + b]) for b in range(self.cols)] for a in range(self.rows)],
                [[int(T[a * self.rows + b]) for b in range(self.rows)] for a in range(self.rows)])

    def encode(self) -> bytes:
        return encode("gfp.witness", {"p": self.p, "count": self.count, "rows": self.rows, "cols": self.cols},
                      pack_entries(self.r, self.p) + pack_entries(self.t, self.p))


@dataclass
class Elements:
    """Per member, a ragged list of F_p elements (roots, or polynomial coefficients)."""
    p: int
    count: int
    offsets: list[int]
    values: object

    def member(self, i: int):
        return [int(x) for x in self.values[self.offsets[i]:self.offsets[i + 1]]]

    def encode(self) -> bytes:
        payload = struct.pack(f"<{len(self.offsets)}Q", *self.offsets) + pack_entries(self.values, self.p)
        return encode("polynomials_fq.elements", {"p": self.p, "count": self.count}, payload)


@dataclass
class Degrees:
    """Per member, a ragged list of naturals (factorisation degrees)."""
    count: int
    offsets: list[int]
    values: list[int]

    def member(self, i: int):
        return [int(x) for x in self.values[self.offsets[i]:self.offsets[i + 1]]]

    def encode(self) -> bytes:
        payload = struct.pack(f"<{len(self.offsets)}Q", *self.offsets) + struct.pack(f"<{len(self.values)}Q", *self.values)
        return encode("polynomials_fq.degrees", {"count": self.count}, payload)


@dataclass
class DegreeSequences:
    count: int
    n: int
    entries: object

    def member(self, i: int):
        return [int(x) for x in self.entries[i * self.n:(i + 1) * self.n]]

    def encode(self) -> bytes:
        return encode("graphs.degree_sequences", {"count": self.count, "n": self.n},
                      pack_entries(self.entries, NATURALS))


@dataclass
class BurnsideCounts:
    values: list[int]

    @property
    def count(self) -> int:
        return len(self.values)

    def member(self, i: int):
        return int(self.values[i])

    def encode(self) -> bytes:
        return encode("burnside.counts", {"count": self.count}, struct.pack(f"<{self.count}Q", *self.values))


@dataclass
class CycleIndex:
    degree: int
    denominator: int
    terms: list[tuple[int, list[int]]]

    @property
    def count(self) -> int:
        return len(self.terms)

    def member(self, i: int):
        multiplicity, cycles = self.terms[i]
        return self.denominator, multiplicity, cycles

    def encode(self) -> bytes:
        words = [word for multiplicity, cycles in self.terms for word in (multiplicity, *cycles)]
        return encode("burnside.cycle_index", {"degree": self.degree, "count": self.count,
                                                "denominator": self.denominator},
                      struct.pack(f"<{len(words)}Q", *words))


@dataclass
class Spectra:
    """Exact character spectra. Each row lists exponents in a sum of powers of zeta_n."""
    n: int
    count: int
    offsets: list[int]
    exponents: object

    def member(self, i: int):
        first = i * self.n
        return [[int(x) for x in self.exponents[self.offsets[first + j]:self.offsets[first + j + 1]]]
                for j in range(self.n)]

    def tolist(self):
        return [self.member(i) for i in range(self.count)]

    def encode(self) -> bytes:
        payload = struct.pack(f"<{len(self.offsets)}Q", *self.offsets) + pack_entries(self.exponents, NATURALS)
        return encode("circulants.spectra", {"n": self.n, "count": self.count}, payload)


@dataclass
class U64Matrices:
    """A batch of fixed-shape natural-number matrices with 64-bit entries."""
    count: int
    rows: int
    cols: int
    entries: object

    def member(self, i: int):
        n = self.rows * self.cols
        flat = self.entries[i * n:(i + 1) * n]
        return [[int(flat[r * self.cols + c]) for c in range(self.cols)] for r in range(self.rows)]

    def tolist(self):
        return [self.member(i) for i in range(self.count)]

    def encode(self) -> bytes:
        payload = struct.pack(f"<{len(self.entries)}Q", *map(int, self.entries))
        return encode("designs.matrix", {"count": self.count, "rows": self.rows, "cols": self.cols}, payload)


@dataclass
class U64Vectors:
    """A batch of fixed-length natural-number vectors with 64-bit entries."""
    count: int
    length: int
    entries: object

    def member(self, i: int):
        return [int(x) for x in self.entries[i * self.length:(i + 1) * self.length]]

    def tolist(self):
        return [self.member(i) for i in range(self.count)]

    def encode(self) -> bytes:
        payload = struct.pack(f"<{len(self.entries)}Q", *map(int, self.entries))
        return encode("polytopes_small.vectors", {"count": self.count, "length": self.length}, payload)


@dataclass
class Partitions:
    count: int
    n: int
    labels: object

    def member(self, i: int):
        return [int(x) for x in self.labels[i * self.n:(i + 1) * self.n]]

    def encode(self) -> bytes:
        return encode("perm_groups.partition", {"count": self.count, "n": self.n},
                      pack_entries(self.labels, 0))


@dataclass
class Bsgs:
    count: int
    n: int
    base_offsets: list[int]
    strong_offsets: list[int]
    bases: object
    strong: object

    def member(self, i: int):
        base = [int(x) for x in self.bases[self.base_offsets[i]:self.base_offsets[i + 1]]]
        generators = []
        for j in range(self.strong_offsets[i], self.strong_offsets[i + 1]):
            generators.append([int(x) for x in self.strong[j * self.n:(j + 1) * self.n]])
        return base, generators

    def encode(self) -> bytes:
        offsets = struct.pack(f"<{len(self.base_offsets)}Q", *self.base_offsets)
        offsets += struct.pack(f"<{len(self.strong_offsets)}Q", *self.strong_offsets)
        return encode("perm_groups.bsgs", {"count": self.count, "n": self.n},
                      offsets + pack_entries(self.bases, 0) + pack_entries(self.strong, 0))


@dataclass
class CharacterTable:
    """One exact cyclotomic character table.

    A cell is the sorted list of exponents of its representative's eigenvalues. With conductor
    `e`, `[a, b]` denotes `zeta_e^a + zeta_e^b`. Spectra are flattened by row, then class.
    """
    order: int
    classes: int
    conductor: int
    representatives: list[int]
    class_sizes: list[int]
    degrees: list[int]
    spectra: object

    @property
    def count(self) -> int:
        return 1

    def row(self, i: int):
        degree = self.degrees[i]
        start = sum(self.degrees[:i]) * self.classes
        return [[int(x) for x in self.spectra[start + c * degree:start + (c + 1) * degree]]
                for c in range(self.classes)]

    def member(self, i: int):
        if i != 0:
            raise IndexError(i)
        return ((self.order, self.conductor), (self.representatives, self.class_sizes),
                (self.degrees, [self.row(r) for r in range(self.classes)]))

    def encode(self) -> bytes:
        if not (len(self.representatives) == len(self.class_sizes) == len(self.degrees) == self.classes):
            raise ValueError("character table class arrays have the wrong length")
        expected = sum(self.degrees) * self.classes
        if len(self.spectra) != expected:
            raise ValueError(f"character table needs {expected} spectral exponents, got {len(self.spectra)}")
        arrays = self.representatives + self.class_sizes + self.degrees
        payload = struct.pack(f"<{len(arrays)}Q", *arrays) + pack_entries(self.spectra, 0)
        return encode("characters.table", {"order": self.order, "classes": self.classes,
                                            "conductor": self.conductor}, payload)


@dataclass
class CharacterIndicators:
    values: list[int]

    @property
    def count(self) -> int:
        return len(self.values)

    def member(self, i: int):
        return int(self.values[i])

    def encode(self) -> bytes:
        return encode("characters.indicators", {"count": self.count}, struct.pack(f"<{self.count}b", *self.values))


@dataclass
class CharacterMultiplicities:
    values: list[int]

    @property
    def count(self) -> int:
        return len(self.values)

    def member(self, i: int):
        return int(self.values[i])

    def encode(self) -> bytes:
        return encode("characters.multiplicities", {"count": self.count},
                      struct.pack(f"<{self.count}Q", *self.values))


@dataclass
class PermutationGenerators:
    """One canonical list of permutation generators per group in a group_tables family."""
    count: int
    order: int
    offsets: list[int]
    entries: object

    def member(self, i: int):
        a, b = self.offsets[i], self.offsets[i + 1]
        return [[int(x) for x in self.entries[j * self.order:(j + 1) * self.order]] for j in range(a, b)]

    def encode(self) -> bytes:
        payload = struct.pack(f"<{len(self.offsets)}Q", *self.offsets) + pack_entries(self.entries, 0)
        return encode("automorphisms.generators", {"count": self.count, "order": self.order}, payload)


@dataclass
class MobiusMatrices:
    """A ragged batch of signed square matrices, one Möbius matrix per poset."""
    count: int
    offsets: list[int]
    entries: list[int]

    def member(self, i: int):
        a, b = self.offsets[i], self.offsets[i + 1]
        n = math.isqrt(b - a)
        if n * n != b - a:
            raise ValueError("posets.mobius member is not square")
        return [[int(self.entries[a + r * n + c]) for c in range(n)] for r in range(n)]

    def encode(self) -> bytes:
        payload = struct.pack(f"<{len(self.offsets)}Q", *self.offsets)
        payload += struct.pack(f"<{len(self.entries)}q", *self.entries)
        return encode("posets.mobius", {"count": self.count}, payload)


@dataclass
class Characters:
    values: list[int]

    @property
    def count(self):
        return len(self.values)

    def member(self, i: int):
        return int(self.values[i])

    def encode(self) -> bytes:
        return encode("young.characters", {"count": self.count}, struct.pack(f"<{self.count}q", *self.values))


@dataclass
class RskPairs:
    count: int
    length: int
    shapes: object
    insertion: object
    recording: object

    def member(self, i: int):
        n = self.length
        shape = [int(x) for x in self.shapes[i * n:(i + 1) * n]]

        def tableau(entries):
            flat = entries[i * n * n:(i + 1) * n * n]
            return [[int(flat[r * n + c]) for c in range(n)] for r in range(n)]

        return shape, tableau(self.insertion), tableau(self.recording)

    def encode(self) -> bytes:
        payload = pack_entries(self.shapes, NATURALS)
        payload += pack_entries(self.insertion, NATURALS)
        payload += pack_entries(self.recording, NATURALS)
        return encode("young.rsk_pairs", {"count": self.count, "length": self.length}, payload)


@dataclass
class CurveGroups:
    """Per member: the invariant factors (n1, n2) of the group of points of an elliptic curve."""
    orders: list[tuple[int, int]]

    @property
    def count(self) -> int:
        return len(self.orders)

    def member(self, i: int):
        n1, n2 = self.orders[i]
        return int(n1), int(n2)

    def encode(self) -> bytes:
        words = [x for pair in self.orders for x in pair]
        return encode("elliptic_curves_fp.group", {"count": self.count},
                      struct.pack(f"<{len(words)}Q", *words))


@dataclass
class Coefficients:
    count: int
    length: int
    values: list[int]

    def member(self, i: int):
        return [int(x) for x in self.values[i * self.length:(i + 1) * self.length]]

    def tolist(self):
        return [self.member(i) for i in range(self.count)]

    def encode(self) -> bytes:
        return encode("graph_polynomials.coefficients", {"count": self.count, "length": self.length},
                      struct.pack(f"<{len(self.values)}q", *self.values))


@dataclass
class StronglyRegularParams:
    """Optional (v, k, lambda, mu) records, one per adjacency matrix."""
    count: int
    present: list[int]
    values: list[int]

    def member(self, i: int):
        return tuple(int(x) for x in self.values[4 * i:4 * i + 4]) if self.present[i] else None

    def encode(self) -> bytes:
        payload = bytes(self.present) + struct.pack(f"<{len(self.values)}Q", *self.values)
        return encode("strongly_regular.params", {"count": self.count}, payload)


@dataclass
class StronglyRegularSpectra:
    """Exact optional SRG spectra. A record `(k, neg, abs_delta, D, f, g)` represents
    `k^1`, `((delta + sqrt(D))/2)^f`, and `((delta - sqrt(D))/2)^g`, where delta is
    `-abs_delta` when `neg` is one and `abs_delta` otherwise."""
    count: int
    present: list[int]
    records: list[tuple[int, int, int, int, int, int]]

    def member(self, i: int):
        return tuple(int(x) for x in self.records[i]) if self.present[i] else None

    def encode(self) -> bytes:
        payload = bytearray(self.present)
        for record in self.records:
            payload.extend(struct.pack("<6Q", *record))
        return encode("strongly_regular.spectra", {"count": self.count}, bytes(payload))


@dataclass
class Integers:
    values: list[int]

    def encode(self) -> bytes:
        return encode("integers", {"count": len(self.values)}, struct.pack(f"<{len(self.values)}Q", *self.values))


@dataclass
class Count:
    value: int
    visited: int
    family_size: int

    def encode(self) -> bytes:
        return encode("count", {"value": self.value, "visited": self.visited, "family_size": self.family_size}, b"")


@dataclass
class Histogram:
    visited: int
    family_size: int
    bins: list[int]

    def encode(self) -> bytes:
        return encode("histogram", {"visited": self.visited, "family_size": self.family_size, "bins": len(self.bins)},
                      struct.pack(f"<{len(self.bins)}Q", *self.bins))


@dataclass
class Hits:
    p: int
    rows: int
    cols: int
    total: int
    visited: int
    family_size: int
    indices: list[int]
    members: Matrix

    def encode(self) -> bytes:
        payload = struct.pack(f"<{len(self.indices)}Q", *self.indices) + pack_entries(self.members.entries, self.p)
        return encode("hits", {"p": self.p, "rows": self.rows, "cols": self.cols, "total": self.total,
                               "visited": self.visited, "family_size": self.family_size,
                               "count": len(self.indices), "materialised": self.members.count}, payload)


@dataclass
class First:
    """`first`: the least true member. `found` is 0 or 1; `member` has `found` matrices."""
    p: int
    rows: int
    cols: int
    found: int
    index: int
    visited: int
    family_size: int
    member: Matrix

    def encode(self) -> bytes:
        return encode("first", {"p": self.p, "rows": self.rows, "cols": self.cols, "found": self.found, "index": self.index,
                                "visited": self.visited, "family_size": self.family_size},
                      pack_entries(self.member.entries, self.p))


@dataclass
class Extremum:
    """`max` / `min`: the extreme value, the least index attaining it, and that member."""
    p: int
    rows: int
    cols: int
    value: int
    index: int
    visited: int
    family_size: int
    member: Matrix

    def encode(self) -> bytes:
        return encode("extremum", {"p": self.p, "rows": self.rows, "cols": self.cols, "value": self.value, "index": self.index,
                                   "visited": self.visited, "family_size": self.family_size},
                      pack_entries(self.member.entries, self.p))


@dataclass
class Family:
    """A family description. `kind` is the family name; `params` its integers; `children` the
    nested objects (matrices and inner families) in payload order."""
    kind: str
    params: dict[str, int]
    children: list = field(default_factory=list)

    def encode(self) -> bytes:
        return encode("family." + self.kind, self.params, b"".join(c.encode() for c in self.children))


KINDS = {"gfp.matrix": Matrix, "orbits.perms": Perms, "graph_iso.groups": GraphGroups,
         "gfp.basis": Basis, "gfp.solutions": Solutions, "gfp.inverses": Inverses,
         "gfp.witness": Witness, "graphs.degree_sequences": DegreeSequences,
         "burnside.counts": BurnsideCounts, "burnside.cycle_index": CycleIndex,
         "characters.table": CharacterTable, "characters.indicators": CharacterIndicators,
         "characters.multiplicities": CharacterMultiplicities, "circulants.spectra": Spectra,
         "designs.matrix": U64Matrices,
         "polytopes_small.vectors": U64Vectors,
         "perm_groups.partition": Partitions, "perm_groups.bsgs": Bsgs,
         "automorphisms.generators": PermutationGenerators, "posets.mobius": MobiusMatrices,
         "young.characters": Characters,
         "young.rsk_pairs": RskPairs, "elliptic_curves_fp.group": CurveGroups,
         "polynomials_fq.elements": Elements, "polynomials_fq.degrees": Degrees,
         "graph_polynomials.coefficients": Coefficients,
         "strongly_regular.params": StronglyRegularParams,
         "strongly_regular.spectra": StronglyRegularSpectra, "integers": Integers, "count": Count,
         "histogram": Histogram, "hits": Hits, "first": First, "extremum": Extremum}


def kind_of(obj) -> str:
    """The interchange kind of a decoded object (`family.<name>` for families)."""
    if isinstance(obj, Family):
        return "family." + obj.kind
    if isinstance(obj, Matrix) and obj.p == NATURALS:
        return "lk.naturals"
    for kind, cls in KINDS.items():
        if isinstance(obj, cls):
            return kind
    raise TypeError(type(obj))


def decode(buf: bytes):
    obj, end = decode_at(buf, 0)
    if end != len(buf):
        raise ValueError("trailing bytes after blob")
    return obj


def decode_at(buf: bytes, offset: int):
    b, end = split(buf, offset)
    k, q, pl = b.kind, b.params, b.payload
    if k == "gfp.matrix":
        n = q["count"] * q["rows"] * q["cols"]
        return Matrix(q["p"], q["count"], q["rows"], q["cols"], unpack_entries(pl, q["p"], n)), end
    if k == "orbits.perms":
        return Perms(q["n"], q["count"], unpack_entries(pl, 0, q["count"] * q["n"])), end
    if k == "graph_iso.groups":
        width = 8 * (q["count"] + 1)
        offsets = list(struct.unpack_from(f"<{q['count'] + 1}Q", pl, 0))
        return GraphGroups(q["count"], q["n"], offsets,
                           unpack_entries(pl[width:], 0, offsets[-1] * q["n"])), end
    if k == "lk.naturals":
        n = q["count"] * q["rows"] * q["cols"]
        return Matrix(NATURALS, q["count"], q["rows"], q["cols"], unpack_entries(pl, NATURALS, n)), end
    if k == "first":
        mem = Matrix(q["p"], q["found"], q["rows"], q["cols"], unpack_entries(pl, q["p"], q["found"] * q["rows"] * q["cols"]))
        return First(q["p"], q["rows"], q["cols"], q["found"], q["index"], q["visited"], q["family_size"], mem), end
    if k == "extremum":
        mem = Matrix(q["p"], 1, q["rows"], q["cols"], unpack_entries(pl, q["p"], q["rows"] * q["cols"]))
        return Extremum(q["p"], q["rows"], q["cols"], q["value"], q["index"], q["visited"], q["family_size"], mem), end
    if k == "gfp.basis":
        offs = list(struct.unpack_from(f"<{q['count'] + 1}Q", pl, 0))
        rest = pl[8 * (q["count"] + 1):]
        return Basis(q["p"], q["count"], q["cols"], offs, unpack_entries(rest, q["p"], offs[-1] * q["cols"])), end
    if k == "gfp.solutions":
        flags = list(pl[:q["count"]])
        return Solutions(q["p"], q["count"], q["length"], flags, unpack_entries(pl[q["count"]:], q["p"], q["count"] * q["length"])), end
    if k == "gfp.inverses":
        flags = list(pl[:q["count"]])
        return Inverses(q["p"], q["count"], q["n"], flags, unpack_entries(pl[q["count"]:], q["p"], q["count"] * q["n"] * q["n"])), end
    if k == "gfp.witness":
        w = entry_width(q["p"])
        nr = q["count"] * q["rows"] * q["cols"]
        nt = q["count"] * q["rows"] * q["rows"]
        return Witness(q["p"], q["count"], q["rows"], q["cols"], unpack_entries(pl[:nr * w], q["p"], nr), unpack_entries(pl[nr * w:], q["p"], nt)), end
    if k == "polynomials_fq.elements":
        offs = list(struct.unpack_from(f"<{q['count'] + 1}Q", pl, 0))
        rest = pl[8 * (q["count"] + 1):]
        return Elements(q["p"], q["count"], offs, unpack_entries(rest, q["p"], offs[-1])), end
    if k == "polynomials_fq.degrees":
        offs = list(struct.unpack_from(f"<{q['count'] + 1}Q", pl, 0))
        vals = list(struct.unpack_from(f"<{offs[-1]}Q", pl, 8 * (q["count"] + 1)))
        return Degrees(q["count"], offs, vals), end
    if k == "graphs.degree_sequences":
        return DegreeSequences(q["count"], q["n"], unpack_entries(pl, NATURALS, q["count"] * q["n"])), end
    if k == "burnside.counts":
        return BurnsideCounts(list(struct.unpack_from(f"<{q['count']}Q", pl, 0))), end
    if k == "burnside.cycle_index":
        width = q["degree"] + 1
        words = struct.unpack_from(f"<{q['count'] * width}Q", pl, 0)
        terms = [(words[i * width], list(words[i * width + 1:(i + 1) * width])) for i in range(q["count"])]
        return CycleIndex(q["degree"], q["denominator"], terms), end
    if k == "circulants.spectra":
        rows = q["count"] * q["n"]
        offset_bytes = 8 * (rows + 1)
        if q["n"] == 0 or q["n"] >= 1 << 32 or len(pl) < offset_bytes:
            raise ValueError("circulants.spectra payload length mismatch")
        offsets = list(struct.unpack_from(f"<{rows + 1}Q", pl, 0))
        if offsets[0] != 0 or any(a > b for a, b in zip(offsets, offsets[1:])):
            raise ValueError("circulants.spectra offsets are not ordered")
        exponents = unpack_entries(pl[offset_bytes:], NATURALS, offsets[-1])
        for row in range(rows):
            values = exponents[offsets[row]:offsets[row + 1]]
            if any(int(e) >= q["n"] for e in values) or any(a > b for a, b in zip(values, values[1:])):
                raise ValueError("circulants.spectra rows must contain sorted exponents below n")
        return Spectra(q["n"], q["count"], offsets, exponents), end
    if k == "designs.matrix":
        n = q["count"] * q["rows"] * q["cols"]
        if len(pl) != n * 8:
            raise ValueError("designs.matrix payload length mismatch")
        return U64Matrices(q["count"], q["rows"], q["cols"], list(struct.unpack_from(f"<{n}Q", pl, 0))), end
    if k == "polytopes_small.vectors":
        n = q["count"] * q["length"]
        if len(pl) != n * 8:
            raise ValueError("polytopes_small.vectors payload length mismatch")
        return U64Vectors(q["count"], q["length"], list(struct.unpack_from(f"<{n}Q", pl, 0))), end
    if k == "perm_groups.partition":
        return Partitions(q["count"], q["n"], unpack_entries(pl, 0, q["count"] * q["n"])), end
    if k == "perm_groups.bsgs":
        count, n = q["count"], q["n"]
        width = 8 * (count + 1)
        base_offsets = list(struct.unpack_from(f"<{count + 1}Q", pl, 0))
        strong_offsets = list(struct.unpack_from(f"<{count + 1}Q", pl, width))
        pos = 2 * width
        base_bytes = base_offsets[-1] * 4
        bases = unpack_entries(pl[pos:pos + base_bytes], 0, base_offsets[-1])
        strong = unpack_entries(pl[pos + base_bytes:], 0, strong_offsets[-1] * n)
        return Bsgs(count, n, base_offsets, strong_offsets, bases, strong), end
    if k == "characters.table":
        classes = q["classes"]
        words = list(struct.unpack_from(f"<{3 * classes}Q", pl, 0))
        representatives = words[:classes]
        class_sizes = words[classes:2 * classes]
        degrees = words[2 * classes:]
        rest = pl[24 * classes:]
        spectra = unpack_entries(rest, 0, sum(degrees) * classes)
        return CharacterTable(q["order"], classes, q["conductor"], representatives, class_sizes, degrees, spectra), end
    if k == "characters.indicators":
        return CharacterIndicators(list(struct.unpack_from(f"<{q['count']}b", pl, 0))), end
    if k == "characters.multiplicities":
        return CharacterMultiplicities(list(struct.unpack_from(f"<{q['count']}Q", pl, 0))), end
    if k == "automorphisms.generators":
        noff = q["count"] + 1
        offsets = list(struct.unpack_from(f"<{noff}Q", pl, 0))
        entries = unpack_entries(pl[8 * noff:], 0, offsets[-1] * q["order"])
        return PermutationGenerators(q["count"], q["order"], offsets, entries), end
    if k == "posets.mobius":
        count = q["count"]
        offset_bytes = 8 * (count + 1)
        if len(pl) < offset_bytes:
            raise ValueError("posets.mobius offsets truncated")
        offsets = list(struct.unpack_from(f"<{count + 1}Q", pl, 0))
        rest = pl[offset_bytes:]
        if offsets[0] != 0 or any(a > b for a, b in zip(offsets, offsets[1:])) or len(rest) != offsets[-1] * 8:
            raise ValueError("posets.mobius invalid offsets or payload length")
        entries = list(struct.unpack_from(f"<{offsets[-1]}q", rest, 0))
        return MobiusMatrices(count, offsets, entries), end
    if k == "young.characters":
        return Characters(list(struct.unpack_from(f"<{q['count']}q", pl, 0))), end
    if k == "young.rsk_pairs":
        n = q["count"] * q["length"]
        square = q["count"] * q["length"] * q["length"]
        shapes = unpack_entries(pl[:4 * n], NATURALS, n)
        insertion = unpack_entries(pl[4 * n:4 * (n + square)], NATURALS, square)
        recording = unpack_entries(pl[4 * (n + square):], NATURALS, square)
        return RskPairs(q["count"], q["length"], shapes, insertion, recording), end
    if k == "elliptic_curves_fp.group":
        words = struct.unpack_from(f"<{2 * q['count']}Q", pl, 0)
        return CurveGroups([(words[2 * i], words[2 * i + 1]) for i in range(q["count"])]), end
    if k == "graph_polynomials.coefficients":
        values = list(struct.unpack_from(f"<{q['count'] * q['length']}q", pl, 0))
        return Coefficients(q["count"], q["length"], values), end
    if k == "strongly_regular.params":
        count = q["count"]
        flags = list(pl[:count])
        values = list(struct.unpack_from(f"<{4 * count}Q", pl, count))
        return StronglyRegularParams(count, flags, values), end
    if k == "strongly_regular.spectra":
        count = q["count"]
        flags = list(pl[:count])
        records = [struct.unpack_from("<6Q", pl, count + 48 * i) for i in range(count)]
        return StronglyRegularSpectra(count, flags, records), end
    if k == "integers":
        return Integers(list(struct.unpack_from(f"<{q['count']}Q", pl, 0))), end
    if k == "count":
        return Count(q["value"], q["visited"], q["family_size"]), end
    if k == "histogram":
        return Histogram(q["visited"], q["family_size"], list(struct.unpack_from(f"<{q['bins']}Q", pl, 0))), end
    if k == "hits":
        idx = list(struct.unpack_from(f"<{q['count']}Q", pl, 0))
        rest = pl[8 * q["count"]:]
        mem = Matrix(q["p"], q["materialised"], q["rows"], q["cols"], unpack_entries(rest, q["p"], q["materialised"] * q["rows"] * q["cols"]))
        return Hits(q["p"], q["rows"], q["cols"], q["total"], q["visited"], q["family_size"], idx, mem), end
    if k.startswith("family."):
        children = []
        pos = 0
        while pos < len(pl):
            child, pos = decode_at(pl, pos)
            children.append(child)
        return Family(k[7:], q, children), end
    raise ValueError(f"unknown object kind {k}")
