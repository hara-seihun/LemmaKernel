"""Direct Python implementation of invariants of Boolean truth tables."""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "python"))
from lemmakernel import naive as rt  # noqa: E402
from lemmakernel.interchange import Family, Matrix, SignedMatrices  # noqa: E402


def walsh(row):
    spectrum = [1 if bit == 0 else -1 for bit in row]
    half = 1
    while half < len(spectrum):
        for block in range(0, len(spectrum), 2 * half):
            for i in range(half):
                a, b = spectrum[block + i], spectrum[block + half + i]
                spectrum[block + i] = a + b
                spectrum[block + half + i] = a - b
        half *= 2
    return spectrum


def nonlinearity(row):
    return (len(row) - max(map(abs, walsh(row)))) // 2


def coordinate_degree(row):
    coefficients = list(row)
    bit = 1
    while bit < len(row):
        for mask in range(len(row)):
            if mask & bit:
                coefficients[mask] ^= coefficients[mask ^ bit]
        bit *= 2
    return max((mask.bit_count() for mask, coefficient in enumerate(coefficients) if coefficient), default=0)


def algebraic_degree(table):
    return max((coordinate_degree(row) for row in table), default=0)


def is_bent(row):
    n = len(row).bit_length() - 1
    return n % 2 == 0 and all(abs(coefficient) == 1 << (n // 2) for coefficient in walsh(row))


def is_apn(table):
    size = len(table[0])
    for a in range(1, size):
        counts = {}
        for x in range(size):
            difference = tuple(row[x] ^ row[x ^ a] for row in table)
            counts[difference] = counts.get(difference, 0) + 1
            if counts[difference] > 2:
                return False
    return True


def linear_orders(size):
    orders = [[0]]
    for _ in range(size.bit_length() - 1):
        extended = []
        for span in orders:
            for image in range(1, size):
                if image not in span:
                    extended.append(span + [x ^ image for x in span])
        orders = extended
    return orders


def affine_class(row):
    orders = linear_orders(len(row))
    return min(
        [row[origin ^ x] for x in order]
        for origin in range(len(row))
        for order in orders
    )


def run(op: str, family: Family, reduction: str = "all", prefix: int | None = None, **args):
    op = op.removeprefix("boolean_functions.")
    members = list(itertools.islice(rt.iter_members(family), prefix))
    p = rt.prime(family)
    rows = len(members[0]) if members else family.params.get("rows", 0)
    cols = len(members[0][0]) if members and rows else family.params.get("cols", 0)
    if p != 2:
        raise ValueError("boolean_functions needs an F_2 matrix family")
    if cols <= 0 or cols & (cols - 1):
        raise ValueError("truth-table column count must be a positive power of two")

    if op in ("nonlinearity", "walsh_spectrum", "is_bent", "affine_class") and rows != 1:
        raise ValueError("operation needs a scalar truth table with one output row")
    if op == "walsh_spectrum":
        spectra = [walsh(table[0]) for table in members]
        return rt.reduce_values(reduction, SignedMatrices(len(spectra), 1, cols,
                                                          [x for spectrum in spectra for x in spectrum]))
    if op == "affine_class":
        forms = [affine_class(table[0]) for table in members]
        return rt.reduce_values(reduction, Matrix(2, len(forms), 1, cols, [x for form in forms for x in form]))
    if op == "nonlinearity":
        return rt.reduce_int(reduction, [nonlinearity(table[0]) for table in members], members, p)
    if op == "algebraic_degree":
        return rt.reduce_int(reduction, [algebraic_degree(table) for table in members], members, p)
    if op == "is_bent":
        return rt.reduce_bool(reduction, [is_bent(table[0]) for table in members], members, p, **args)
    if op == "is_apn":
        return rt.reduce_bool(reduction, [is_apn(table) for table in members], members, p, **args)
    raise ValueError(f"unknown operation {op}")
