"""Direct finite-difference baseline for concise circuit fires."""
from __future__ import annotations

import itertools
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "python"))
from lemmakernel import naive as rt  # noqa: E402
from lemmakernel.interchange import Family, Solutions  # noqa: E402


def rank(rows, p):
    matrix = [list(map(lambda value: value % p, row)) for row in rows]
    if not matrix:
        return 0
    output = 0
    for column in range(len(matrix[0])):
        pivot = next((row for row in range(output, len(matrix)) if matrix[row][column]), None)
        if pivot is None:
            continue
        matrix[output], matrix[pivot] = matrix[pivot], matrix[output]
        scale = pow(matrix[output][column], p - 2, p)
        matrix[output] = [value * scale % p for value in matrix[output]]
        for row in range(len(matrix)):
            if row != output and matrix[row][column]:
                coefficient = matrix[row][column]
                matrix[row] = [
                    (left - coefficient * right) % p
                    for left, right in zip(matrix[row], matrix[output])
                ]
        output += 1
    return output


def solve(rows, rhs, p):
    augmented = [list(row) + [value] for row, value in zip(rows, rhs)]
    variables = len(rows[0])
    pivots = []
    output = 0
    for column in range(variables + 1):
        pivot = next((row for row in range(output, len(augmented)) if augmented[row][column]), None)
        if pivot is None:
            continue
        augmented[output], augmented[pivot] = augmented[pivot], augmented[output]
        scale = pow(augmented[output][column], p - 2, p)
        augmented[output] = [value * scale % p for value in augmented[output]]
        for row in range(len(augmented)):
            if row != output and augmented[row][column]:
                coefficient = augmented[row][column]
                augmented[row] = [
                    (left - coefficient * right) % p
                    for left, right in zip(augmented[row], augmented[output])
                ]
        pivots.append(column)
        output += 1
    if variables in pivots:
        return None
    solution = [0] * variables
    for row, column in enumerate(pivots):
        solution[column] = augmented[row][variables]
    return solution


def normalized(vector, p):
    first = next((value for value in vector if value), None)
    if first is None:
        return None
    scale = pow(first, p - 2, p)
    return tuple(value * scale % p for value in vector)


def is_circuit(configuration, p, base_dim):
    if not configuration or not 0 < base_dim < len(configuration[0]):
        return False
    directions = [row[:base_dim] for row in configuration]
    covectors = [row[base_dim:] for row in configuration]
    if any(normalized(row, p) is None for row in directions + covectors):
        return False
    if len({normalized(row, p) for row in directions}) != len(directions):
        return False
    if rank(directions, p) != base_dim or rank(covectors, p) != len(covectors[0]):
        return False
    tensors = [
        [u * z % p for u in covector for z in direction]
        for direction, covector in zip(directions, covectors)
    ]
    if any(sum(row[column] for row in tensors) % p for column in range(len(tensors[0]))):
        return False
    return rank(tensors, p) + 1 == len(tensors)


def add(left, right, p):
    return tuple((x + y) % p for x, y in zip(left, right))


def field_points(p, dimension):
    return [tuple(index // p**coordinate % p for coordinate in range(dimension))
            for index in range(p**dimension)]


def equation_rows(configuration, p, base_dim):
    directions = [tuple(row[:base_dim]) for row in configuration]
    covectors = [row[base_dim:] for row in configuration]
    fibre_dim = len(covectors[0])
    points = field_points(p, base_dim)
    index = {point: position for position, point in enumerate(points)}
    potential_columns = len(points) * fibre_dim
    rows = []
    for circuit_row, (direction, covector) in enumerate(zip(directions, covectors)):
        for point in points:
            row = [0] * (potential_columns + len(configuration))
            shifted = index[add(point, direction, p)]
            current = index[point]
            for coordinate, value in enumerate(covector):
                row[shifted * fibre_dim + coordinate] = (
                    row[shifted * fibre_dim + coordinate] + value
                ) % p
                row[current * fibre_dim + coordinate] = (
                    row[current * fibre_dim + coordinate] - value
                ) % p
            row[potential_columns + circuit_row] = p - 1
            rows.append(row)
    for coordinate in range(fibre_dim):
        row = [0] * (potential_columns + len(configuration))
        row[coordinate] = 1
        rows.append(row)
    return rows


def is_fire(configuration, p, base_dim):
    if not is_circuit(configuration, p, base_dim):
        return False
    equations = equation_rows(configuration, p, base_dim)
    carry = [0] * (len(equations[0]) - len(configuration)) + [1] * len(configuration)
    return rank(equations + [carry], p) > rank(equations, p)


def find_potential(configuration, p, base_dim):
    if not is_circuit(configuration, p, base_dim):
        return None
    equations = equation_rows(configuration, p, base_dim)
    potential_columns = p**base_dim * (len(configuration[0]) - base_dim)
    carry = [0] * potential_columns + [1] * len(configuration)
    return solve(equations + [carry], [0] * len(equations) + [1], p)


def verifies(configuration, p, base_dim, defects, potential):
    if not is_circuit(configuration, p, base_dim):
        return False
    fibre_dim = len(configuration[0]) - base_dim
    points = field_points(p, base_dim)
    if len(defects) != len(configuration) or len(potential) != len(points):
        return False
    if any(len(value) != fibre_dim for value in potential) or any(potential[0]):
        return False
    index = {point: position for position, point in enumerate(points)}
    for circuit_row, row in enumerate(configuration):
        direction, covector = tuple(row[:base_dim]), row[base_dim:]
        for point in points:
            shifted = potential[index[add(point, direction, p)]]
            current = potential[index[point]]
            value = sum(u * (left - right) for u, left, right in zip(covector, shifted, current)) % p
            if value != defects[circuit_row] % p:
                return False
    return sum(defects) % p != 0


def run(op: str, family: Family, reduction: str = "all", prefix: int | None = None, **args):
    op = op.removeprefix("circuit_fires.")
    if family.kind != "explicit":
        raise ValueError("circuit_fires operations need an explicit family")
    batch = family.children[0]
    p = batch.p
    if p > 7 or p < 2 or any(p % divisor == 0 for divisor in range(2, math.isqrt(p) + 1)):
        raise ValueError("circuit_fires needs a prime field of order at most 7")
    base_dim = args["base_dim"]
    members = list(itertools.islice(rt.iter_members(family), prefix))
    if not 0 < base_dim < batch.cols:
        raise ValueError("base_dim must be positive and smaller than the member column count")
    if op == "is_circuit":
        values = [is_circuit(member, p, base_dim) for member in members]
    elif op == "is_fire":
        values = [is_fire(member, p, base_dim) for member in members]
    elif op == "find_potential":
        fibre_dim = batch.cols - base_dim
        variables = p**base_dim * fibre_dim + batch.rows
        equations = batch.rows * p**base_dim + fibre_dim + 1
        if variables > 1024 or equations * (variables + 1) > 2**24:
            raise ValueError("configuration is too large for explicit potential construction")
        if reduction != "all":
            raise ValueError("find_potential values only reduce with `all`")
        solutions = [find_potential(member, p, base_dim) for member in members]
        return Solutions(
            p, len(members), variables,
            [int(solution is not None) for solution in solutions],
            [value for solution in solutions for value in (solution or [0] * variables)],
        )
    elif op == "verifies_potential":
        defects_object, potential_object = args["defects"], args["potential"]
        defects = defects_object.member(0)[0]
        potential = potential_object.member(0)
        if defects_object.p != p or defects_object.count != 1 or defects_object.rows != 1 or defects_object.cols != batch.rows:
            raise ValueError("defects must have one entry per configuration row over the same prime")
        if potential_object.p != p or potential_object.count != 1 or potential_object.rows != p**base_dim or potential_object.cols != batch.cols - base_dim:
            raise ValueError("potential must have p^base_dim rows and fibre_dim columns over the same prime")
        values = [verifies(member, p, base_dim, defects, potential) for member in members]
    else:
        raise ValueError(f"unknown operation {op}")
    return rt.reduce_bool(reduction, values, members, p, **args)
