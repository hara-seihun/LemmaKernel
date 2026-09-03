"""Direct fixed-size CI classification for finite group tables."""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "python"))
from lemmakernel import naive as rt  # noqa: E402
from lemmakernel.interchange import Family  # noqa: E402


def identity_of(table):
    n = len(table)
    return next(e for e in range(n)
                if all(table[e][x] == x and table[x][e] == x for x in range(n)))


def inverses_of(table, identity):
    n = len(table)
    return [next(y for y in range(n)
                 if table[x][y] == identity and table[y][x] == identity)
            for x in range(n)]


def connection_sets(table, k):
    identity = identity_of(table)
    inverse = inverses_of(table, identity)
    atoms = []
    seen = {identity}
    for x in range(len(table)):
        if x in seen:
            continue
        atom = tuple(sorted({x, inverse[x]}))
        atoms.append(atom)
        seen.update(atom)
    sets = []
    for choices in itertools.product((False, True), repeat=len(atoms)):
        selected = tuple(x for chosen, atom in zip(choices, atoms) if chosen for x in atom)
        if len(selected) == k:
            sets.append(selected)
    return sets, inverse


def preserves_table(table, permutation):
    n = len(table)
    return all(permutation[table[a][b]] == table[permutation[a]][permutation[b]]
               for a in range(n) for b in range(n))


def automorphisms(table):
    return [permutation for permutation in itertools.permutations(range(len(table)))
            if preserves_table(table, permutation)]


def canonical_aut_key(selected, group_automorphisms):
    return min(tuple(sorted(automorphism[x] for x in selected))
               for automorphism in group_automorphisms)


def cayley_graph(table, inverse, selected):
    selected = set(selected)
    n = len(table)
    return [[int(x != y and table[inverse[x]][y] in selected) for y in range(n)]
            for x in range(n)]


def canonical_graph_key(adjacency):
    n = len(adjacency)
    return min(tuple(adjacency[order[i]][order[j]] for i in range(n) for j in range(n))
               for order in itertools.permutations(range(n)))


def class_counts(table, k):
    sets, inverse = connection_sets(table, k)
    group_automorphisms = automorphisms(table)
    aut_keys = {canonical_aut_key(selected, group_automorphisms) for selected in sets}
    iso_keys = {canonical_graph_key(cayley_graph(table, inverse, selected)) for selected in sets}
    return len(aut_keys), len(iso_keys)


def run(op: str, family: Family, reduction: str = "all", prefix: int | None = None, **args):
    op = op.removeprefix("cayley_iso.")
    if family.kind != "group_tables":
        raise ValueError("cayley_iso operations need a group_tables family")
    tables = list(itertools.islice(rt.iter_members(family), prefix))
    counts = [class_counts(table, args["k"]) for table in tables]

    if op == "aut_class_count":
        return rt.reduce_int(reduction, [aut for aut, _ in counts], tables, rt.NATURALS)
    if op == "iso_class_count":
        return rt.reduce_int(reduction, [iso for _, iso in counts], tables, rt.NATURALS)
    if op in ("is_ci", "is_non_ci"):
        flags = [aut == iso for aut, iso in counts]
        if op == "is_non_ci":
            flags = [not flag for flag in flags]
        return rt.reduce_bool(reduction, flags, tables, rt.NATURALS, **args)
    raise ValueError(f"unknown operation {op}")
