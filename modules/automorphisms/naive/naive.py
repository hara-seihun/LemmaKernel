"""The direct automorphisms implementation.

Every permutation of a Cayley table's labels is tested against every product. The generic backend
uses propagated partial maps; this file stays deliberately literal for the benchmark baseline.
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "python"))
from lemmakernel import naive as rt  # noqa: E402
from lemmakernel.interchange import Family, PermutationGenerators  # noqa: E402


def preserves(table, permutation):
    n = len(table)
    return all(permutation[table[a][b]] == table[permutation[a]][permutation[b]]
               for a in range(n) for b in range(n))


def automorphisms(table):
    return [list(f) for f in itertools.permutations(range(len(table))) if preserves(table, f)]


def compose(a, b):
    return [b[a[x]] for x in range(len(a))]


def generated_group(n, generators):
    identity = tuple(range(n))
    seen = {identity}
    queue = [list(identity)]
    for a in queue:
        for g in generators:
            b = compose(a, g)
            key = tuple(b)
            if key not in seen:
                seen.add(key)
                queue.append(b)
    return seen


def canonical_generators(table, autos=None):
    autos = autos if autos is not None else automorphisms(table)
    generators = []
    subgroup = generated_group(len(table), generators)
    for a in autos:
        if tuple(a) not in subgroup:
            generators.append(a)
            subgroup = generated_group(len(table), generators)
    return generators


def center_size(table):
    n = len(table)
    return sum(all(table[a][b] == table[b][a] for b in range(n)) for a in range(n))


def run(op: str, family: Family, reduction: str = "all", prefix: int | None = None, **_):
    op = op.removeprefix("automorphisms.")
    if family.kind != "group_tables":
        raise ValueError("automorphisms operations need a group_tables family")
    tables = list(itertools.islice(rt.iter_members(family), prefix))
    autos = [automorphisms(table) for table in tables]

    if op == "aut_generators":
        per_group = [canonical_generators(table, group_autos) for table, group_autos in zip(tables, autos)]
        offsets = [0]
        entries = []
        for generators in per_group:
            offsets.append(offsets[-1] + len(generators))
            entries.extend(x for generator in generators for x in generator)
        order = len(tables[0]) if tables else family.params["rows"]
        return rt.reduce_values(reduction, PermutationGenerators(len(tables), order, offsets, entries))

    if op == "aut_order":
        values = [len(group_autos) for group_autos in autos]
    elif op == "holomorph_order":
        values = [len(table) * len(group_autos) for table, group_autos in zip(tables, autos)]
    elif op == "inner_aut_index":
        values = [len(group_autos) * center_size(table) // len(table)
                  for table, group_autos in zip(tables, autos)]
    else:
        raise ValueError(f"unknown operation {op}")
    return rt.reduce_int(reduction, values, tables, rt.NATURALS)
