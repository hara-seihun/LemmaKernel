"""Readable whole-family algorithms for finite block designs."""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "python"))
from lemmakernel import naive as rt  # noqa: E402
from lemmakernel.interchange import Family, Perms, U64Matrices  # noqa: E402


def output(rows):
    rows = [list(map(int, row)) for row in rows]
    cols = len(rows[0]) if rows else 0
    if any(len(row) != cols for row in rows):
        raise ValueError("ragged designs result")
    return U64Matrices(1, len(rows), cols, [x for row in rows for x in row])


def parse_blocks(family: Family, prefix: int | None = None):
    members = list(itertools.islice(rt.iter_members(family), prefix))
    if not members:
        raise ValueError("a block family must contain at least one block")
    k, v = len(members[0]), len(members[0][0])
    if not (1 <= k <= v):
        raise ValueError("blocks must have shape k x v with 1 <= k <= v")
    blocks = []
    for member in members:
        block = []
        if len(member) != k:
            raise ValueError("block shape changed within the family")
        for row in member:
            points = [i for i, x in enumerate(row) if x == 1]
            if len(row) != v or len(points) != 1 or any(x not in (0, 1) for x in row):
                raise ValueError("block rows must be distinct standard basis vectors in increasing point order")
            block.append(points[0])
        if any(a >= b for a, b in zip(block, block[1:])):
            raise ValueError("block rows must be distinct standard basis vectors in increasing point order")
        blocks.append(tuple(block))
    return v, k, blocks


def subsets(v, size):
    return list(itertools.combinations(range(v), size))


def lambda_vector(v, blocks, t):
    return [sum(set(s).issubset(block) for block in blocks) for s in subsets(v, t)]


def intersection_numbers(k, blocks):
    counts = [0] * (k + 1)
    for a, b in itertools.combinations(blocks, 2):
        counts[len(set(a) & set(b))] += 1
    return counts


def parallel_classes(v, k, blocks):
    class_size = v // k
    points = set(range(v))
    return [c for c in itertools.combinations(range(len(blocks)), class_size)
            if set().union(*(set(blocks[i]) for i in c)) == points
            and sum(len(blocks[i]) for i in c) == v]


def resolution(v, k, blocks):
    if v % k or not blocks:
        return []
    class_size = v // k
    if not class_size or len(blocks) % class_size:
        return []
    classes = parallel_classes(v, k, blocks)

    def search(remaining):
        if not remaining:
            return []
        first = min(remaining)
        for candidate in classes:
            if first in candidate and set(candidate) <= remaining:
                rest = search(remaining - set(candidate))
                if rest is not None:
                    return [list(candidate), *rest]
        return None

    return search(set(range(len(blocks)))) or []


def act_subset(g, subset):
    return tuple(sorted(g[i] for i in subset))


def subset_orbits(v, size, generators):
    universe = subsets(v, size)
    remaining = set(universe)
    orbits = []
    for representative in universe:
        if representative not in remaining:
            continue
        seen = {representative}
        queue = [representative]
        for item in queue:
            for generator in generators:
                image = act_subset(generator, item)
                if image not in seen:
                    seen.add(image)
                    queue.append(image)
        remaining -= seen
        orbits.append(queue)
    return orbits


def kramer_mesner(family: Family, t: int, group):
    if family.kind != "subsets":
        raise ValueError("kramer_mesner_matrix needs a subsets family")
    if not isinstance(group, Perms):
        raise ValueError("group must be a permutation group on the dictionary positions")
    v = len(rt.dictionary(family))
    k = family.params["k"]
    if not 0 <= t <= k:
        raise ValueError("t must satisfy 0 <= t <= k")
    generators = group.tolist()
    if not generators or group.n != v:
        raise ValueError("permutation generators must act on every dictionary position")
    row_orbits = subset_orbits(v, t, generators)
    col_orbits = subset_orbits(v, k, generators)
    return [[sum(set(row_orbit[0]).issubset(block) for block in col_orbit)
             for col_orbit in col_orbits]
            for row_orbit in row_orbits]


def run(op: str, family: Family, reduction: str = "all", prefix: int | None = None, **args):
    if reduction != "all":
        raise ValueError("designs.matrix values only reduce with `all`")
    op = op.removeprefix("designs.")
    if op == "kramer_mesner_matrix":
        return output(kramer_mesner(family, args["t"], args["group"]))

    v, k, blocks = parse_blocks(family, prefix)
    if op in ("is_design", "lambda_vector"):
        t = args["t"]
        if not 0 <= t <= k:
            raise ValueError("t must satisfy 0 <= t <= k")
        counts = lambda_vector(v, blocks, t)
        if op == "lambda_vector":
            return output([counts])
        return output([[int(all(x == counts[0] for x in counts)), counts[0]]])
    if op == "intersection_numbers":
        return output([intersection_numbers(k, blocks)])
    if op == "dual_is_design":
        replications = [sum(point in block for block in blocks) for point in range(v)]
        intersections = [len(set(a) & set(b)) for a, b in itertools.combinations(blocks, 2)]
        flag = (len(blocks) >= 2 and replications[0] >= 2
                and len(set(replications)) == 1 and len(set(intersections)) == 1)
        return output([[int(flag), intersections[0] if intersections else 0]])
    if op == "is_resolvable":
        return output(resolution(v, k, blocks))
    raise ValueError(f"unknown operation {op}")
