"""Obvious exact implementation of small lattice-polytope operations.

Every family member is materialised. Determinants enumerate supporting facets, facet
intersections enumerate faces, and Ehrhart values enumerate every lattice point in each small
dilated bounding box.
"""
from __future__ import annotations

import itertools
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "python"))
from lemmakernel import naive as rt  # noqa: E402
from lemmakernel.interchange import Family, U64Vectors  # noqa: E402


def determinant(matrix):
    if not matrix:
        return 1
    row = matrix[0]
    total = 0
    for j, value in enumerate(row):
        minor = [r[:j] + r[j + 1:] for r in matrix[1:]]
        total += (-1 if j & 1 else 1) * value * determinant(minor)
    return total


def matrix_rank(matrix):
    if not matrix:
        return 0
    rows, cols = len(matrix), len(matrix[0])
    for k in range(min(rows, cols), 0, -1):
        for rs in itertools.combinations(range(rows), k):
            for cs in itertools.combinations(range(cols), k):
                if determinant([[matrix[i][j] for j in cs] for i in rs]):
                    return k
    return 0


def affine_rank(points):
    if not points:
        return 0
    base = points[0]
    return matrix_rank([[x - y for x, y in zip(point, base)] for point in points[1:]])


def unique(points):
    return [list(point) for point in dict.fromkeys(map(tuple, points))]


def projection(points, rank, dimension):
    for cols in itertools.combinations(range(dimension), rank):
        if affine_rank([[point[j] for j in cols] for point in points]) == rank:
            return cols
    return tuple(range(rank))


def analyse(points):
    points = unique(points)
    dimension = len(points[0])
    rank = affine_rank(points)
    cols = projection(points, rank, dimension)
    projected = [[point[j] for j in cols] for point in points]
    facets = {}
    if rank:
        for choice in itertools.combinations(range(len(points)), rank):
            selected = [projected[i] for i in choice]
            if affine_rank(selected) != rank - 1:
                continue
            base = selected[0]
            differences = [[x - y for x, y in zip(point, base)] for point in selected[1:]]
            normal = [(-1 if j & 1 else 1) * determinant([row[:j] + row[j + 1:] for row in differences])
                      for j in range(rank)]
            constant = sum(x * y for x, y in zip(normal, base))
            values = [sum(x * y for x, y in zip(normal, point)) - constant for point in projected]
            if all(value <= 0 for value in values) and any(values):
                pass
            elif all(value >= 0 for value in values) and any(values):
                normal = [-x for x in normal]
                constant = -constant
                values = [-x for x in values]
            else:
                continue
            indices = tuple(i for i, value in enumerate(values) if value == 0)
            facets.setdefault(indices, (normal, constant))
    faces = {tuple(range(len(points)))}
    for indices in facets:
        current = list(faces)
        face_set = set(indices)
        for face in current:
            intersection = tuple(i for i in face if i in face_set)
            if intersection:
                faces.add(intersection)
    return {
        "points": points,
        "dimension": dimension,
        "rank": rank,
        "projection": cols,
        "projected": projected,
        "facets": [(indices, *plane) for indices, plane in facets.items()],
        "faces": sorted(faces),
    }


def face_dimension(analysis, face):
    return affine_rank([analysis["projected"][i] for i in face])


def f_vector(analysis):
    return [sum(face_dimension(analysis, face) == k for face in analysis["faces"])
            for k in range(analysis["dimension"] + 1)]


def is_simplicial(analysis):
    rank = analysis["rank"]
    if rank <= 1:
        return True
    vertices = {face[0] for face in analysis["faces"] if face_dimension(analysis, face) == 0}
    return all(sum(i in vertices for i in facet) == rank for facet, _, _ in analysis["facets"])


def lattice_count(analysis, t):
    if t == 0:
        return 1
    points = analysis["points"]
    scaled = [[t * x for x in point] for point in points]
    bounds = [range(t * min(point[j] for point in points), t * max(point[j] for point in points) + 1)
              for j in range(analysis["dimension"])]
    count = 0
    for point in itertools.product(*bounds):
        if affine_rank(scaled + [list(point)]) != analysis["rank"]:
            continue
        projected = [point[j] for j in analysis["projection"]]
        if all(sum(x * y for x, y in zip(normal, projected)) <= t * constant
               for _, normal, constant in analysis["facets"]):
            count += 1
    return count


def h_star(analysis):
    rank = analysis["rank"]
    counts = [lattice_count(analysis, t) for t in range(rank + 1)]
    hs = []
    for j in range(rank + 1):
        hs.append(sum((-1) ** (j - i) * math.comb(rank + 1, j - i) * counts[i]
                      for i in range(j + 1)))
    return hs + [0] * (analysis["dimension"] + 1 - len(hs))


def run(op: str, family: Family, reduction: str = "all", prefix: int | None = None, **args):
    op = op.removeprefix("polytopes_small.")
    if family.kind != "subsets":
        raise ValueError("polytopes_small operations are defined on subsets families only")
    dictionary = rt.dictionary(family)
    if family.params["k"] == 0 or not dictionary:
        raise ValueError("polytope members must be nonempty")
    dimension = len(dictionary[0])
    if dimension > 6:
        raise ValueError("ambient dimension must be at most 6")
    members = list(itertools.islice(rt.iter_members(family), prefix))
    p = rt.prime(family)
    if op == "vertex_count":
        values = []
        for member in members:
            points = unique(member)
            values.append(len(points) if all(x in (0, 1) for point in points for x in point)
                          else f_vector(analyse(points))[0])
        return rt.reduce_int(reduction, values, members, p)
    analyses = [analyse(member) for member in members]
    if op == "is_simplicial":
        return rt.reduce_bool(reduction, [is_simplicial(a) for a in analyses], members, p, **args)
    if reduction != "all":
        raise ValueError(f"{op} values only reduce with `all`")
    if op == "f_vector":
        values = [f_vector(a) for a in analyses]
    elif op == "ehrhart_polynomial":
        values = [h_star(a) for a in analyses]
    else:
        raise ValueError(f"unknown operation {op}")
    return U64Vectors(len(values), dimension + 1, [x for value in values for x in value])
