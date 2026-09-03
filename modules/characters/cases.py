"""Oracle cases for exact abelian character tables."""
from __future__ import annotations

import cmath
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.harness import Case, cyclic, symmetric  # noqa: E402


def translations(moduli):
    points = []

    def add_points(prefix):
        if len(prefix) == len(moduli):
            points.append(prefix)
            return
        for value in range(moduli[len(prefix)]):
            add_points(prefix + (value,))

    add_points(())
    index = {point: i for i, point in enumerate(points)}
    generators = []
    for coordinate, modulus in enumerate(moduli):
        generator = []
        for point in points:
            image = list(point)
            image[coordinate] = (image[coordinate] + 1) % modulus
            generator.append(index[tuple(image)])
        generators.append(generator)
    return generators


def cases(ctx, rng):
    del rng
    groups = [
        ("trivial", ctx.group_elements(ctx.perms(1, cyclic(1)))),
        ("C2", ctx.group_elements(ctx.perms(2, cyclic(2)))),
        ("C3", ctx.group_elements(ctx.perms(3, cyclic(3)))),
        ("C4", ctx.group_elements(ctx.perms(4, cyclic(4)))),
        ("V4", ctx.group_elements(ctx.perms(4, translations([2, 2])))),
    ]
    out = []
    for name, group in groups:
        out.append(Case(f"{name} table", group, "character_table"))
        out.append(Case(f"{name} indicators", group, "frobenius_schur"))

    c4 = groups[3][1]
    c2_in_c4 = ctx.perms(4, [[2, 3, 0, 1]])
    for character in (0, 1):
        out.append(Case(f"C4 character {character} restricted to C2", c4, "restrict",
                        {"subgroup": c2_in_c4, "character": character}))
        out.append(Case(f"C2 character {character} induced to C4", c4, "induce",
                        {"subgroup": c2_in_c4, "character": character}))

    s3 = ctx.group_elements(ctx.perms(3, symmetric(3)))
    out += [
        Case("S3 table", s3, "character_table", oracle=False),
        Case("S3 indicators", s3, "frobenius_schur", oracle=False),
        Case("range is not a group", ctx.range(0, 3), "character_table", oracle=False),
        Case("subgroup is not contained", c4, "restrict",
             {"subgroup": ctx.perms(4, [[1, 0, 2, 3]]), "character": 0}, oracle=False),
        Case("character index is outside the table", c4, "induce",
             {"subgroup": c2_in_c4, "character": 99}, oracle=False),
    ]
    return out


def invariants(ctx):
    group = ctx.group_elements(ctx.perms(16, translations([8, 2])))
    table = ctx.value("characters.character_table", group)
    indicators = ctx.value("characters.frobenius_schur", group)
    subgroup = ctx.perms(16, [translations([8, 2])[1]])
    restricted = ctx.value("characters.restrict", group, subgroup=subgroup, character=1)
    induced = ctx.value("characters.induce", group, subgroup=subgroup, character=0)
    assert sum(restricted.values) == 1
    assert sum(induced.values) == 8
    assert table.order == table.classes == 16
    assert table.representatives == list(range(16))
    assert table.class_sizes == [1] * 16
    assert table.degrees == [1] * 16
    assert table.row(0) == [[0]] * 16
    assert len({tuple(cell[0] for cell in table.row(i)) for i in range(16)}) == 16

    roots = [[sum(cmath.exp(2j * math.pi * exponent / table.conductor) for exponent in cell)
              for cell in table.row(i)] for i in range(table.classes)]
    for i in range(table.classes):
        for j in range(table.classes):
            inner = sum(roots[i][k] * roots[j][k].conjugate() for k in range(table.classes))
            assert abs(inner - (table.order if i == j else 0)) < 1e-8
        average = sum(value * value for value in roots[i]) / table.order
        assert abs(average - indicators.values[i]) < 1e-8
