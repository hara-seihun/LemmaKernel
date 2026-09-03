"""Plain Python character tables for finite abelian permutation groups.

The implementation materialises the group and its multiplication table, splits it as a direct
product of cyclic groups, and enumerates every homomorphism to roots of unity. It shares no state
with the C++ backend.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "python"))
from lemmakernel import naive as rt  # noqa: E402
from lemmakernel.interchange import (CharacterIndicators, CharacterMultiplicities, CharacterTable,
                                     Family, Perms)  # noqa: E402

MAX_ORDER = 4096


def _group(family: Family):
    if family.kind != "group_elements":
        raise ValueError("characters operations are defined on group_elements families only")
    (generators,) = family.children
    elements = rt.perm_closure(generators.tolist())
    if len(elements) > MAX_ORDER:
        raise ValueError(f"group has more than {MAX_ORDER} elements")
    index = {tuple(g): i for i, g in enumerate(elements)}
    identity = index[tuple(range(generators.n))]
    multiplication = [[index[tuple(rt.compose(a, b))] for b in elements] for a in elements]
    if any(multiplication[i][j] != multiplication[j][i]
           for i in range(len(elements)) for j in range(i)):
        raise ValueError("character_table generic currently accepts abelian groups only")

    orders = []
    exponent = 1
    for i in range(len(elements)):
        power = identity
        order = 0
        while True:
            power = multiplication[power][i]
            order += 1
            if power == identity:
                break
            if order > len(elements):
                raise RuntimeError("element order does not divide the group order")
        orders.append(order)
        exponent = math.lcm(exponent, order)
    return elements, identity, multiplication, orders, exponent


def _decomposition(identity, multiplication, orders):
    size = len(multiplication)

    def search(subgroup, coordinates, factors):
        if len(subgroup) == size:
            return factors, coordinates
        inside = set(subgroup)
        candidates = sorted((x for x in range(size) if x not in inside), key=lambda x: (-orders[x], x))
        for generator in candidates:
            cyclic_order = orders[generator]
            powers = [identity]
            for _ in range(1, cyclic_order):
                powers.append(multiplication[powers[-1]][generator])
            if any(x in inside for x in powers[1:]):
                continue
            extended = []
            next_coordinates = list(coordinates)
            seen = set()
            for h in subgroup:
                for k, power in enumerate(powers):
                    x = multiplication[h][power]
                    if x in seen:
                        break
                    seen.add(x)
                    extended.append(x)
                    next_coordinates[x] = coordinates[h] + (k,)
                else:
                    continue
                break
            else:
                answer = search(extended, next_coordinates, factors + (cyclic_order,))
                if answer is not None:
                    return answer
        return None

    answer = search([identity], [()] * size, ())
    if answer is None:
        raise RuntimeError("could not split the finite abelian group into cyclic factors")
    return answer


def _dual_rows(size, exponent, factors, coordinates):
    rows = []

    def visit(prefix):
        if len(prefix) != len(factors):
            for value in range(factors[len(prefix)]):
                visit(prefix + (value,))
            return
        rows.append([sum(r * k * (exponent // order)
                         for r, k, order in zip(prefix, coordinates[element], factors)) % exponent
                     for element in range(size)])

    visit(())
    rows.sort()
    if len(rows) != size or len({tuple(row) for row in rows}) != size:
        raise RuntimeError("dual-group enumeration produced the wrong number of characters")
    return rows


def _compute(family):
    elements, identity, multiplication, orders, exponent = _group(family)
    factors, coordinates = _decomposition(identity, multiplication, orders)
    return elements, exponent, _dual_rows(len(elements), exponent, factors, coordinates)


def _same_restriction(ambient, subgroup, ambient_row, subgroup_row):
    ambient_elements, ambient_conductor, _ = ambient
    subgroup_elements, subgroup_conductor, _ = subgroup
    if ambient_conductor % subgroup_conductor:
        return False
    index = {tuple(element): i for i, element in enumerate(ambient_elements)}
    scale = ambient_conductor // subgroup_conductor
    return all(tuple(element) in index and ambient_row[index[tuple(element)]] == subgroup_row[i] * scale % ambient_conductor
               for i, element in enumerate(subgroup_elements))


def run(op: str, family: Family, reduction: str = "all", prefix: int | None = None, **args):
    del prefix
    op = op.removeprefix("characters.")
    if reduction != "all":
        raise ValueError(f"{op} values only reduce with `all`")
    ambient = _compute(family)
    elements, conductor, rows = ambient
    order = len(elements)
    if op == "character_table":
        return CharacterTable(order, order, conductor, list(range(order)), [1] * order, [1] * order,
                              [exponent for row in rows for exponent in row])
    if op == "frobenius_schur":
        return CharacterIndicators([int(all(2 * exponent % conductor == 0 for exponent in row)) for row in rows])
    if op in ("restrict", "induce"):
        generators = args.get("subgroup")
        if not isinstance(generators, Perms):
            raise ValueError("subgroup must be a permutation group")
        subgroup = _compute(Family("group_elements", {}, [generators]))
        if generators.n != len(elements[0]):
            raise ValueError("subgroup permutations must have the same degree as the ambient group")
        if not set(map(tuple, subgroup[0])) <= set(map(tuple, elements)):
            raise ValueError("subgroup is not contained in the ambient group")
        character = args["character"]
        source = rows if op == "restrict" else subgroup[2]
        targets = subgroup[2] if op == "restrict" else rows
        if not 0 <= character < len(source):
            raise ValueError("character index is outside the table")
        values = [_same_restriction(ambient, subgroup, source[character], target) if op == "restrict"
                  else _same_restriction(ambient, subgroup, target, source[character]) for target in targets]
        return CharacterMultiplicities([int(value) for value in values])
    raise ValueError(f"unknown operation {op}")
