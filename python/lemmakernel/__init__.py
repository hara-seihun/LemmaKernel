"""LemmaKernel Python binding: a thin ctypes layer over the C ABI in runtime/include/lemmakernel/lk.h.

    import lemmakernel as lk
    ctx = lk.Context()                         # every backend, every core
    G = ctx.grassmannian(p=2, n=8, h=4)        # a family: all 4-dim subspaces of F_2^8
    F = ctx.transform(G, lk.matrix(2, C))      # each basis mapped through C
    h = ctx.run("gfp.rank", F, "histogram")    # a Handle; .value() decodes it
    print(h.value().bins)

Families, operations and reductions are whatever `describe()` says the library has.
"""
from __future__ import annotations

import ctypes
import json
import os
from pathlib import Path

from . import interchange
from .interchange import (NATURALS, Basis, BurnsideCounts, Count, CycleIndex, Extremum, Family, First, Histogram, Hits,
                          Integers, Inverses, Matrix, Perms, Solutions, Witness, matrix, naturals, perms)
from ._manifest import MODULES

__all__ = ["Context", "Handle", "Error", "describe", "matrix", "perms", "naturals", "interchange", "MODULES", "Perms",
           "Matrix", "Basis", "Solutions", "Inverses", "Witness", "BurnsideCounts", "CycleIndex", "Integers", "Count",
           "Histogram", "Hits", "First", "Extremum", "Family", "NATURALS"]


class Error(RuntimeError):
    def __init__(self, status: int, message: str):
        super().__init__(f"[{ {1: 'invalid argument', 2: 'unsupported', 3: 'out of memory', 4: 'internal'}.get(status, status)}] {message}")
        self.status = status


def _find_library() -> str:
    env = os.environ.get("LEMMAKERNEL_LIB")
    if env:
        return env
    here = Path(__file__).resolve().parent
    for cand in (here / "liblemmakernel.so", here.parent.parent / "build" / "liblemmakernel.so"):
        if cand.exists():
            return str(cand)
    import ctypes.util
    found = ctypes.util.find_library("lemmakernel")
    if found:
        return found
    raise ImportError("liblemmakernel.so not found; build it (cmake -B build && ninja -C build) or set LEMMAKERNEL_LIB")


class _Arg(ctypes.Structure):
    _fields_ = [("name", ctypes.c_char_p), ("handle", ctypes.c_uint64), ("integer", ctypes.c_uint64)]


_lib = ctypes.CDLL(_find_library())
_H = ctypes.c_uint64
_P = ctypes.c_void_p
_lib.lk_context_create.argtypes = [ctypes.c_char_p, ctypes.POINTER(_P)]
_lib.lk_context_destroy.argtypes = [_P]
_lib.lk_context_set_threads.argtypes = [_P, ctypes.c_uint32]
_lib.lk_context_error.argtypes = [_P]
_lib.lk_context_error.restype = ctypes.c_char_p
_lib.lk_describe.restype = ctypes.c_char_p
_lib.lk_import.argtypes = [_P, ctypes.c_char_p, ctypes.c_size_t, ctypes.POINTER(_H)]
_lib.lk_export.argtypes = [_P, _H, ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_size_t)]
_lib.lk_free.argtypes = [ctypes.c_void_p]
_lib.lk_release.argtypes = [_P, _H]
_lib.lk_handle_kind.argtypes = [_P, _H, ctypes.POINTER(ctypes.c_char_p)]
_lib.lk_handle_param.argtypes = [_P, _H, ctypes.c_char_p, ctypes.POINTER(ctypes.c_uint64)]
_lib.lk_family_explicit.argtypes = [_P, _H, ctypes.POINTER(_H)]
_lib.lk_family_subsets.argtypes = [_P, _H, ctypes.c_uint64, ctypes.POINTER(_H)]
_lib.lk_family_grassmannian.argtypes = [_P, ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64, ctypes.POINTER(_H)]
_lib.lk_family_all_matrices.argtypes = [_P, ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64, ctypes.POINTER(_H)]
_lib.lk_family_transform.argtypes = [_P, _H, _H, ctypes.POINTER(_H)]
_lib.lk_family_stack.argtypes = [_P, _H, _H, ctypes.POINTER(_H)]
_lib.lk_family_group_elements.argtypes = [_P, _H, ctypes.POINTER(_H)]
_lib.lk_family_subsets_of.argtypes = [_P, _H, ctypes.c_uint64, ctypes.POINTER(_H)]
_lib.lk_family_symmetric_matrices.argtypes = [_P, ctypes.c_uint64, ctypes.c_uint64, ctypes.POINTER(_H)]
_lib.lk_family_range.argtypes = [_P, ctypes.c_uint64, ctypes.c_uint64, ctypes.POINTER(_H)]
_lib.lk_family_words.argtypes = [_P, ctypes.c_uint64, ctypes.c_uint64, ctypes.POINTER(_H)]
_lib.lk_family_size.argtypes = [_P, _H, ctypes.POINTER(ctypes.c_uint64)]
_lib.lk_family_member.argtypes = [_P, _H, ctypes.c_uint64, ctypes.POINTER(_H)]
_lib.lk_run.argtypes = [_P, ctypes.c_char_p, _H, ctypes.c_char_p, ctypes.POINTER(_Arg), ctypes.c_size_t, ctypes.POINTER(_H)]


def describe() -> dict:
    """Everything this library build contains, plus which backends run on this machine."""
    return json.loads(_lib.lk_describe().decode())


class Handle:
    """An object living inside a Context. Released when garbage collected or on .release()."""

    def __init__(self, ctx: "Context", h: int):
        self._ctx = ctx
        self._h = h

    def __repr__(self):
        return f"<lk.Handle {self.kind} {self.params}>"

    @property
    def kind(self) -> str:
        out = ctypes.c_char_p()
        self._ctx._check(_lib.lk_handle_kind(self._ctx._ptr, self._h, ctypes.byref(out)))
        return out.value.decode()

    def param(self, name: str) -> int:
        out = ctypes.c_uint64()
        self._ctx._check(_lib.lk_handle_param(self._ctx._ptr, self._h, name.encode(), ctypes.byref(out)))
        return out.value

    @property
    def params(self) -> dict[str, int]:
        kind = self.kind
        names = _PARAM_NAMES.get(kind)
        if names is None:
            if kind.startswith("family."):
                names = ["p", "rows", "cols", "size"]
            else:
                names = []
        out = {}
        for n in names:
            try:
                out[n] = self.param(n)
            except Error:
                pass
        return out

    def export(self) -> bytes:
        buf = ctypes.c_void_p()
        n = ctypes.c_size_t()
        self._ctx._check(_lib.lk_export(self._ctx._ptr, self._h, ctypes.byref(buf), ctypes.byref(n)))
        try:
            return ctypes.string_at(buf, n.value)
        finally:
            _lib.lk_free(buf)

    def value(self):
        """Decode into the Python object for this kind (see lemmakernel.interchange)."""
        return interchange.decode(self.export())

    def release(self):
        if self._h and self._ctx._ptr:
            _lib.lk_release(self._ctx._ptr, self._h)
            self._h = 0

    def __del__(self):
        try:
            self.release()
        except Exception:
            pass


_PARAM_NAMES = {
    "gfp.matrix": ["p", "count", "rows", "cols"], "gfp.basis": ["p", "count", "cols"], "orbits.perms": ["n", "count"],
    "gfp.solutions": ["p", "count", "length"], "gfp.inverses": ["p", "count", "n"],
    "gfp.witness": ["p", "count", "rows", "cols"], "burnside.counts": ["count"],
    "burnside.cycle_index": ["degree", "count", "denominator"], "integers": ["count"],
    "count": ["value", "visited", "family_size"], "histogram": ["visited", "family_size", "bins"],
    "hits": ["p", "rows", "cols", "total", "visited", "family_size", "count", "materialised"],
    "lk.naturals": ["count", "rows", "cols"],
    "first": ["p", "rows", "cols", "found", "index", "visited", "family_size"],
    "extremum": ["p", "rows", "cols", "value", "index", "visited", "family_size"],
}


class Context:
    """Owns handles and picks backends. `backend` is None/"auto" or a name like "gfp.generic"."""

    def __init__(self, backend: str | None = None, threads: int | None = None):
        self._ptr = None
        ptr = _P()
        st = _lib.lk_context_create(backend.encode() if backend else None, ctypes.byref(ptr))
        if st:
            raise Error(st, f"cannot create context for backend {backend!r}; see describe()['available_backends']")
        self._ptr = ptr
        if threads is not None:
            self.threads = threads

    def close(self):
        if self._ptr:
            _lib.lk_context_destroy(self._ptr)
            self._ptr = None

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    @property
    def threads(self) -> int | None:
        return self._threads if hasattr(self, "_threads") else None

    @threads.setter
    def threads(self, n: int):
        self._check(_lib.lk_context_set_threads(self._ptr, n))
        self._threads = n

    def _check(self, status: int):
        if status:
            raise Error(status, (_lib.lk_context_error(self._ptr) or b"").decode())

    def _wrap(self, h: _H) -> Handle:
        return Handle(self, h.value)

    # objects
    def load(self, blob: bytes) -> Handle:
        out = _H()
        self._check(_lib.lk_import(self._ptr, blob, len(blob), ctypes.byref(out)))
        return self._wrap(out)

    def put(self, obj) -> Handle:
        """Import an interchange object (Matrix, Family, ...)."""
        return self.load(obj.encode())

    def matrix(self, p: int, data, rows: int | None = None, cols: int | None = None) -> Handle:
        return self.put(matrix(p, data, rows, cols))

    def perms(self, n: int, data) -> Handle:
        return self.put(perms(n, data))

    def naturals(self, data, rows: int | None = None, cols: int | None = None) -> Handle:
        return self.put(naturals(data, rows, cols))

    # families
    def explicit(self, batch) -> Handle:
        out = _H()
        b = self._keep(batch)
        self._check(_lib.lk_family_explicit(self._ptr, b._h, ctypes.byref(out)))
        return self._wrap(out)

    def subsets(self, dictionary, k: int) -> Handle:
        out = _H()
        d = self._keep(dictionary)
        self._check(_lib.lk_family_subsets(self._ptr, d._h, k, ctypes.byref(out)))
        return self._wrap(out)

    def grassmannian(self, p: int, n: int, h: int) -> Handle:
        out = _H()
        self._check(_lib.lk_family_grassmannian(self._ptr, p, n, h, ctypes.byref(out)))
        return self._wrap(out)

    def all_matrices(self, p: int, rows: int, cols: int) -> Handle:
        out = _H()
        self._check(_lib.lk_family_all_matrices(self._ptr, p, rows, cols, ctypes.byref(out)))
        return self._wrap(out)

    def transform(self, family, matrix) -> Handle:
        out = _H()
        f, m = self._keep(family), self._keep(matrix)
        self._check(_lib.lk_family_transform(self._ptr, f._h, m._h, ctypes.byref(out)))
        return self._wrap(out)

    def stack(self, family, rows) -> Handle:
        out = _H()
        f, m = self._keep(family), self._keep(rows)
        self._check(_lib.lk_family_stack(self._ptr, f._h, m._h, ctypes.byref(out)))
        return self._wrap(out)

    def group_elements(self, generators) -> Handle:
        out = _H()
        g = self._keep(generators)
        self._check(_lib.lk_family_group_elements(self._ptr, g._h, ctypes.byref(out)))
        return self._wrap(out)

    def subsets_of(self, family, k: int) -> Handle:
        """k-subsets of another family's members, each flattened to one row."""
        out = _H()
        f = self._keep(family)
        self._check(_lib.lk_family_subsets_of(self._ptr, f._h, k, ctypes.byref(out)))
        return self._wrap(out)

    def symmetric_matrices(self, p: int, n: int) -> Handle:
        out = _H()
        self._check(_lib.lk_family_symmetric_matrices(self._ptr, p, n, ctypes.byref(out)))
        return self._wrap(out)

    def range(self, a: int, b: int) -> Handle:
        """The integers a <= x < b as 1 x 1 natural-number matrices."""
        out = _H()
        self._check(_lib.lk_family_range(self._ptr, a, b, ctypes.byref(out)))
        return self._wrap(out)

    def words(self, alphabet: int, length: int) -> Handle:
        """Every word of `length` letters over 0..alphabet-1 as a 1 x length natural-number matrix."""
        out = _H()
        self._check(_lib.lk_family_words(self._ptr, alphabet, length, ctypes.byref(out)))
        return self._wrap(out)

    def size(self, family) -> int:
        out = ctypes.c_uint64()
        f = self._keep(family)
        self._check(_lib.lk_family_size(self._ptr, f._h, ctypes.byref(out)))
        return out.value

    def member(self, family, index: int) -> Handle:
        out = _H()
        f = self._keep(family)
        self._check(_lib.lk_family_member(self._ptr, f._h, index, ctypes.byref(out)))
        return self._wrap(out)

    def _keep(self, x) -> Handle:
        if isinstance(x, Handle):
            return x
        if hasattr(x, "encode"):
            return self.put(x)
        raise TypeError(f"expected a Handle or an interchange object, got {type(x).__name__}")

    # operations
    def run(self, op: str, family, reduction: str = "all", **args) -> Handle:
        """Run `op` (e.g. "gfp.rank") over `family` with `reduction`. Keyword arguments are the
        named arguments of the operation and reduction: handles/objects or integers."""
        fam = self._keep(family)
        kept = [fam]  # temporaries must outlive the call
        cargs = (_Arg * max(1, len(args)))()
        names = [n.encode() for n in args]
        for i, (name, v) in enumerate(args.items()):
            cargs[i].name = names[i]
            if isinstance(v, int) and not isinstance(v, bool):
                cargs[i].handle = 0
                cargs[i].integer = v
            else:
                h = self._keep(v)
                kept.append(h)
                cargs[i].handle = h._h
                cargs[i].integer = 0
        out = _H()
        self._check(_lib.lk_run(self._ptr, op.encode(), fam._h, reduction.encode(), cargs, len(args), ctypes.byref(out)))
        del kept
        return self._wrap(out)

    def value(self, op: str, family, reduction: str = "all", **args):
        """run() and decode the result."""
        return self.run(op, family, reduction, **args).value()
