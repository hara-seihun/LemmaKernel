"""The runtime's concepts, in plain Python, for a module's naive implementation.

A naive implementation only has to know its own mathematics: it materialises the family with
`members`, computes one value per member, and hands the values to `reduce_int`, `reduce_bool`,
or `reduce_values`. Families and reductions are runtime concepts, so their naive form lives
here, once, and matches the runtime's canonical orders and the Lean reference exactly.

    members(family)          -> (list of members as lists of rows, p)
    dictionary(family)       -> the rows a subsets/subsets_of family draws from
    reduce_int(reduction, values, members, p)          -> interchange object
    reduce_bool(reduction, flags, members, p, limit=0) -> interchange object
    reduce_values(reduction, obj)                      -> obj (`all` only)

`p` is the field-size tag of the members, 0 for permutations, and NATURALS for integer members.
"""
from __future__ import annotations

import functools
import itertools

from .interchange import NATURALS, Count, Extremum, Family, First, Histogram, Hits, Integers, Matrix, Perms


# ---- arithmetic the families need ---------------------------------------------------------------

def matmul(a, b, p):
    return [[sum(x * y for x, y in zip(row, col)) % p for col in zip(*b)] for row in a]


def compose(g, h):
    """Permutation x -> h(g(x))."""
    return [h[g[x]] for x in range(len(g))]


def perm_closure(gens: list[list[int]]) -> list[list[int]]:
    """Every element of the generated permutation group, sorted lexicographically."""
    n = len(gens[0])
    identity = list(range(n))
    seen = {tuple(identity)}
    queue = [identity]
    for g in queue:
        for gen in gens:
            h = compose(g, gen)
            if tuple(h) not in seen:
                seen.add(tuple(h))
                queue.append(h)
    return sorted(queue)


def matrix_closure(gens: list[list[list[int]]], p: int) -> list[list[list[int]]]:
    """Every element of the generated matrix group, sorted by flattened entries."""
    n = len(gens[0])
    identity = [[int(i == j) for j in range(n)] for i in range(n)]
    seen = {tuple(map(tuple, identity))}
    queue = [identity]
    for a in queue:
        for gen in gens:
            b = matmul(a, gen, p)
            key = tuple(map(tuple, b))
            if key not in seen:
                seen.add(key)
                queue.append(b)
    return sorted(queue)


# ---- families -----------------------------------------------------------------------------------

def batch_members(m: Matrix | Perms):
    if isinstance(m, Perms):
        return [[m.member(i)] for i in range(m.count)]
    return [m.member(i) for i in range(m.count)]


def vectors_of(m):
    """A batch of 1 x n rows, one k x n matrix, or a permutation batch as vectors."""
    if isinstance(m, Perms):
        return m.tolist()
    return [r[0] for r in batch_members(m)] if m.rows == 1 else m.member(0)


def grassmannian_members(p, n, h):
    """Every field-labelled RREF basis shape: pivot sets in lexicographic order, then free
    entries in row-major lexicographic order."""
    for piv in itertools.combinations(range(n), h):
        free = [(i, c) for i in range(h) for c in range(piv[i] + 1, n) if c not in piv]
        for digits in itertools.product(range(p), repeat=len(free)):
            rows = [[0] * n for _ in range(h)]
            for i, pc in enumerate(piv):
                rows[i][pc] = 1
            for (i, c), d in zip(free, digits):
                rows[i][c] = d
            yield rows


def symmetric_members(p, n):
    """Every symmetric n x n matrix over an alphabet of size p. The upper triangle, row-major,
    contains the base-p digits of the index."""
    upper = [(i, j) for i in range(n) for j in range(i, n)]
    for digits in itertools.product(range(p), repeat=len(upper)):
        m = [[0] * n for _ in range(n)]
        for (i, j), d in zip(upper, digits):
            m[i][j] = m[j][i] = d
        yield m


def partition_members(total, max_part, max_parts, max_multiplicity, distinct, odd):
    """Constrained partitions, descending lexicographic, padded to `total` entries."""
    upper = min(max_part or total, total)
    slots = min(max_parts or total, total)
    cap = 1 if distinct else max_multiplicity or total
    values = [v for v in range(upper, 0, -1) if not odd or v % 2]

    def rec(i, remaining, left, prefix):
        if remaining == 0:
            yield [prefix + [0] * (total - len(prefix))]
            return
        if i == len(values) or left == 0:
            return
        value = values[i]
        for count in range(min(cap, left, remaining // value), -1, -1):
            yield from rec(i + 1, remaining - count * value, left - count,
                           prefix + [value] * count)

    yield from rec(0, total, slots, [])


def composition_members(total, parts, max_part):
    """Positive bounded compositions, by length then descending lexicographic."""
    maximum = min(max_part or total, total)

    def exact(remaining, left, prefix):
        if left == 0:
            if remaining == 0:
                yield [prefix + [0] * (total - len(prefix))]
            return
        for value in range(min(maximum, remaining), 0, -1):
            yield from exact(remaining - value, left - 1, prefix + [value])

    for length in ([parts] if parts else range(1, total + 1)):
        yield from exact(total, length, [])


def standard_tableaux(shape):
    """Standard tableaux ordered by the row of each removable corner, top row first."""
    shape = list(shape)
    rows, cols, label = len(shape), shape[0], sum(shape)
    tableau = [[0] * cols for _ in range(rows)]

    def fill(current, value):
        if value == 0:
            yield [list(row) for row in tableau]
            return
        for i, width in enumerate(current):
            below = current[i + 1] if i + 1 < rows else 0
            if width == below:
                continue
            col = width - 1
            current[i] -= 1
            tableau[i][col] = value
            yield from fill(current, value - 1)
            tableau[i][col] = 0
            current[i] += 1

    yield from fill(shape, label)


def graph_edges(n):
    return [(i, j) for i in range(n) for j in range(i + 1, n)]


def graph_from_mask(mask, n):
    edges = graph_edges(n)
    a = [[0] * n for _ in range(n)]
    for q, (i, j) in enumerate(edges):
        a[i][j] = a[j][i] = (mask >> (len(edges) - 1 - q)) & 1
    return a


def graph_mask(a):
    mask = 0
    for i, j in graph_edges(len(a)):
        mask = (mask << 1) | int(bool(a[i][j]))
    return mask


def canonical_graph(a):
    """Lexicographically least adjacency matrix, using ordered-partition individualisation."""
    n = len(a)
    assigned = []
    best = None

    def search(cells):
        nonlocal best
        if len(assigned) == n:
            candidate = [[a[assigned[i]][assigned[j]] for j in range(n)] for i in range(n)]
            flat = [x for row in candidate for x in row]
            if best is None or flat < best[0]:
                best = flat, candidate
            return
        first = cells[0]
        tried = []
        for v in first:
            if any(all(w in (u, v) or a[u][w] == a[v][w] for cell in cells for w in cell) for u in tried):
                continue
            tried.append(v)
            assigned.append(v)
            refined = []
            for cell in cells:
                zero = [w for w in cell if w != v and not a[v][w]]
                one = [w for w in cell if w != v and a[v][w]]
                if zero:
                    refined.append(zero)
                if one:
                    refined.append(one)
            search(refined)
            assigned.pop()

    search([list(range(n))])
    return best[1]


@functools.lru_cache(maxsize=None)
def all_graph_masks(n):
    if n == 1:
        return (0,)
    found = set()
    for parent_mask in all_graph_masks(n - 1):
        parent = graph_from_mask(parent_mask, n - 1)
        for neighbourhood in range(1 << (n - 1)):
            a = [[0] * n for _ in range(n)]
            for i in range(n - 1):
                a[i][:n - 1] = parent[i]
                a[i][n - 1] = a[n - 1][i] = (neighbourhood >> (n - 2 - i)) & 1
            found.add(graph_mask(canonical_graph(a)))
    return tuple(sorted(found))


def inverse_classes(elements):
    identity = list(range(len(elements[0])))
    used = {tuple(identity)}
    classes = []
    for g in elements:
        if tuple(g) in used:
            continue
        inverse = next(h for h in elements if compose(g, h) == identity)
        classes.append([g] if inverse == g else [g, inverse])
        used.add(tuple(g))
        used.add(tuple(inverse))
    return classes


def cayley_graph_members(gens):
    elements = perm_closure(gens)
    classes = inverse_classes(elements)
    index = {tuple(g): i for i, g in enumerate(elements)}
    for bits in itertools.product(range(2), repeat=len(classes)):
        a = [[0] * len(elements) for _ in elements]
        selected = [s for bit, cls in zip(bits, classes) if bit for s in cls]
        for i, g in enumerate(elements):
            for s in selected:
                j = index[tuple(compose(g, s))]
                a[i][j] = a[j][i] = 1
        yield a


def prime(f: Family) -> int:
    if f.kind in ("group_tables", "range", "words", "partitions", "compositions", "standard_tableaux"):
        return NATURALS
    if f.kind == "group_elements":
        return f.children[0].p if isinstance(f.children[0], Matrix) else 0
    if f.kind in ("all_graphs", "edge_subgraphs", "cayley_graphs"):
        return 2
    if "p" in f.params:
        return f.params["p"]
    child = f.children[0]
    return prime(child) if isinstance(child, Family) else getattr(child, "p", 0)


def dictionary(f: Family) -> list[list[int]]:
    """The rows a `subsets` family chooses from, or the flattened members of a `subsets_of`
    family's inner family."""
    if f.kind == "subsets":
        (d,) = f.children
        return vectors_of(d)
    if f.kind == "subsets_of":
        (inner,) = f.children
        return [[x for r in m for x in r] for m in iter_members(inner)]
    raise ValueError(f"{f.kind} has no dictionary")


def iter_members(f: Family):
    """Yield the members of a family in canonical order as lists of rows."""
    if f.kind in ("explicit", "group_tables"):
        (batch,) = f.children
        yield from batch_members(batch)
    elif f.kind in ("subsets", "subsets_of"):
        vecs = dictionary(f)
        for idx in itertools.combinations(range(len(vecs)), f.params["k"]):
            yield [vecs[i] for i in idx]
    elif f.kind == "grassmannian":
        yield from grassmannian_members(f.params["p"], f.params["n"], f.params["h"])
    elif f.kind == "all_matrices":
        p, rows, cols = f.params["p"], f.params["rows"], f.params["cols"]
        for digits in itertools.product(range(p), repeat=rows * cols):
            yield [list(digits[r * cols:(r + 1) * cols]) for r in range(rows)]
    elif f.kind == "symmetric_matrices":
        yield from symmetric_members(f.params["p"], f.params["n"])
    elif f.kind == "transform":
        inner, C = f.children
        c, p = C.member(0), C.p
        for m in iter_members(inner):
            yield matmul(m, c, p)
    elif f.kind == "stack":
        inner, rows = f.children
        extra = rows.member(0)
        for m in iter_members(inner):
            yield m + extra
    elif f.kind == "group_elements":
        (gens,) = f.children
        if hasattr(gens, "p"):
            yield from matrix_closure(gens.tolist(), gens.p)
        else:
            for g in perm_closure(gens.tolist()):
                yield [g]
    elif f.kind == "range":
        for x in range(f.params["a"], f.params["b"]):
            yield [[x]]
    elif f.kind == "words":
        for w in itertools.product(range(f.params["alphabet"]), repeat=f.params["length"]):
            yield [list(w)]
    elif f.kind == "partitions":
        yield from partition_members(f.params["total"], f.params["max_part"], f.params["max_parts"],
                                     f.params["max_multiplicity"], f.params["distinct"], f.params["odd"])
    elif f.kind == "compositions":
        yield from composition_members(f.params["total"], f.params["parts"], f.params["max_part"])
    elif f.kind == "standard_tableaux":
        (shape_obj,) = f.children
        shape = shape_obj.member(0)[0]
        yield from standard_tableaux(shape)
    elif f.kind == "all_graphs":
        for mask in all_graph_masks(f.params["n"]):
            yield graph_from_mask(mask, f.params["n"])
    elif f.kind == "edge_subgraphs":
        (host,) = f.children
        a = host.member(0)
        edges = [(i, j) for i, j in graph_edges(len(a)) if a[i][j]]
        for chosen in itertools.combinations(edges, f.params["k"]):
            g = [[0] * len(a) for _ in a]
            for i, j in chosen:
                g[i][j] = g[j][i] = 1
            yield g
    elif f.kind == "cayley_graphs":
        (gens,) = f.children
        yield from cayley_graph_members(gens.tolist())
    else:
        raise ValueError(f"unknown family {f.kind}")


def members(f: Family):
    """Materialise a family in canonical order as a list of matrices (lists of rows), plus p."""
    return list(iter_members(f)), prime(f)


# ---- reductions ---------------------------------------------------------------------------------

def _shape(members, p):
    rows = len(members[0]) if members else 0
    cols = len(members[0][0]) if members else 0
    return rows, cols


def _one(p, rows, cols, m):
    return Matrix(p, 1, rows, cols, [x for r in m for x in r])


def reduce_int(reduction: str, values: list[int], members, p: int):
    """Integer values of `members` (in family order) under an integer reduction."""
    size = len(values)
    rows, cols = _shape(members, p)
    if reduction == "all":
        return Integers(values)
    if reduction == "histogram":
        bins = [0] * (max(values) + 1 if values else 0)
        for v in values:
            bins[v] += 1
        return Histogram(size, size, bins)
    if reduction == "sum":
        return Count(sum(values), size, size)
    if reduction in ("max", "min"):
        if not values:
            raise ValueError(f"{reduction} of an empty family")
        v = max(values) if reduction == "max" else min(values)
        i = values.index(v)
        return Extremum(p, rows, cols, v, i, size, size, _one(p, rows, cols, members[i]))
    raise ValueError(f"reduction {reduction} does not accept integer values")


def reduce_bool(reduction: str, flags: list[bool], members, p: int, limit: int = 0, **_):
    """Boolean values of `members` (in family order) under a boolean reduction."""
    size = len(flags)
    rows, cols = _shape(members, p)
    if reduction == "all":
        return Integers([int(f) for f in flags])
    if reduction == "count":
        return Count(sum(flags), size, size)
    if reduction == "hits":
        idx = [i for i, f in enumerate(flags) if f]
        n = min(limit, len(idx))
        mem = Matrix(p, n, rows, cols, [x for i in idx[:n] for r in members[i] for x in r])
        return Hits(p, rows, cols, len(idx), size, size, idx, mem)
    if reduction == "first":
        i = next((i for i, f in enumerate(flags) if f), None)
        if i is None:
            return First(p, rows, cols, 0, 0, size, size, Matrix(p, 0, rows, cols, []))
        return First(p, rows, cols, 1, i, i + 1, size, _one(p, rows, cols, members[i]))
    raise ValueError(f"reduction {reduction} does not accept boolean values")


def reduce_values(reduction: str, obj):
    """A materialised per-member value only reduces with `all`."""
    if reduction != "all":
        raise ValueError(f"{type(obj).__name__} values only reduce with `all`")
    return obj
