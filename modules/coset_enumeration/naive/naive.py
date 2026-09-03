"""Plain deterministic Todd-Coxeter enumeration for the coset_enumeration module."""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "python"))
from lemmakernel import naive as rt  # noqa: E402
from lemmakernel.interchange import CosetRepresentations, Family, Matrix, NATURALS  # noqa: E402

NONE = -1


class Table:
    def __init__(self, generators: int, bound: int):
        self.generators = generators
        self.letters = 2 * generators
        self.bound = bound
        self.table: list[list[int]] = [[NONE] * self.letters]
        self.parent = [0]
        self.failed = False

    def root(self, coset: int) -> int:
        while self.parent[coset] != coset:
            coset = self.parent[coset]
        return coset

    def cell(self, coset: int, letter: int) -> int:
        value = self.table[self.root(coset)][letter]
        return NONE if value == NONE else self.root(value)

    def add_coset(self) -> int:
        if len(self.parent) >= self.bound:
            self.failed = True
            return NONE
        coset = len(self.parent)
        self.parent.append(coset)
        self.table.append([NONE] * self.letters)
        return coset

    def process(self, tasks: list[tuple[str, int, int, int]]) -> None:
        steps = 0
        budget = 1024 + 16 * (self.bound + 1) ** 2 * (self.letters + 1)
        while tasks and not self.failed:
            steps += 1
            if steps > budget:
                self.failed = True
                return
            kind, a, b, d = tasks.pop(0)
            if kind == "merge":
                ra, rb = self.root(a), self.root(b)
                if ra == rb:
                    continue
                lo, hi = sorted((ra, rb))
                self.parent[hi] = lo
                copied = [("link", lo, letter, value)
                          for letter, value in enumerate(self.table[hi]) if value != NONE]
                tasks = copied + tasks
                continue

            letter = b
            rc, rd = self.root(a), self.root(d)
            forward = self.cell(rc, letter)
            backward = self.cell(rd, letter ^ 1)
            if forward != NONE and forward != rd:
                tasks = [("merge", forward, rd, 0), ("link", rc, letter, rd)] + tasks
            elif backward != NONE and backward != rc:
                tasks = [("merge", backward, rc, 0), ("link", rc, letter, rd)] + tasks
            else:
                self.table[rc][letter] = rd
                self.table[rd][letter ^ 1] = rc

    def enforce(self, start: int, word: list[int], end: int) -> None:
        fuel = self.bound + len(word) + 2
        while fuel and not self.failed:
            fuel -= 1
            c, i = self.root(start), 0
            while i < len(word):
                nxt = self.cell(c, word[i])
                if nxt == NONE:
                    break
                c, i = nxt, i + 1

            d, j = self.root(end), len(word)
            while j > i:
                prev = self.cell(d, word[j - 1] ^ 1)
                if prev == NONE:
                    break
                d, j = prev, j - 1

            gap = j - i
            if gap == 0:
                self.process([("merge", c, d, 0)])
                return
            if gap == 1:
                self.process([("link", c, word[i], d)])
                return
            new = self.add_coset()
            if self.failed:
                return
            self.process([("link", c, word[i], new)])
        if not self.failed:
            self.failed = True

    def enumerate(self, relators: list[list[int]], subgroup: list[list[int]]):
        for word in subgroup:
            self.enforce(0, word, 0)
        cursor = 0
        while cursor < len(self.parent) and not self.failed:
            if self.root(cursor) == cursor:
                for word in relators:
                    self.enforce(cursor, word, cursor)
                    if self.failed:
                        break
                for generator in range(self.generators):
                    c = self.root(cursor)
                    if self.cell(c, 2 * generator) == NONE:
                        new = self.add_coset()
                        if self.failed:
                            break
                        self.process([("link", c, 2 * generator, new)])
            cursor += 1
        if self.failed:
            return None

        roots = [i for i in range(len(self.parent)) if self.root(i) == i]
        positions = {c: i for i, c in enumerate(roots)}
        permutations = []
        for generator in range(self.generators):
            permutations.append([positions[self.cell(c, 2 * generator)] for c in roots])
        return permutations


def _words(matrix, letters: int):
    if not isinstance(matrix, Matrix) or matrix.p != NATURALS:
        raise ValueError("subgroup must be natural-number vectors")
    words = []
    for raw in rt.vectors_of(matrix):
        word = []
        padding = False
        for value in map(int, raw):
            if value < letters and not padding:
                word.append(value)
            elif value == letters:
                padding = True
            else:
                raise ValueError("subgroup word contains an invalid letter or non-trailing padding")
        words.append(word)
    return words


def _relation_relators(matrix, letters: int):
    if not isinstance(matrix, Matrix) or matrix.p != NATURALS:
        raise ValueError("relations must be an lk.naturals matrix of encoded equations")
    relators = []
    for raw in rt.vectors_of(matrix):
        row = [int(value) for value in raw]
        if len(row) < 2:
            raise ValueError("relation row needs two lengths")
        left_length, right_length = row[:2]
        if left_length + right_length > len(row) - 2:
            raise ValueError("relation lengths exceed the row width")
        left = row[2:2 + left_length]
        right = row[2 + left_length:2 + left_length + right_length]
        if any(letter >= letters for letter in left + right):
            raise ValueError("relation has a symbol outside the generator alphabet")
        if any(value != letters for value in row[2 + left_length + right_length:]):
            raise ValueError("relation padding must equal the alphabet size")
        relators.append(left + [letter ^ 1 for letter in reversed(right)])
    return relators


def _setup(family: Family, generators: int, relations, max_cosets: int, subgroup=None):
    if family.kind != "words":
        raise ValueError("coset enumeration is defined on words families only")
    if generators < 1 or generators >= 1 << 31:
        raise ValueError("generators must satisfy 1 <= generators < 2^31")
    if family.params["alphabet"] != 2 * generators:
        raise ValueError("words alphabet must equal 2*generators")
    if max_cosets < 1 or max_cosets >= 1 << 32:
        raise ValueError("max_cosets must satisfy 1 <= max_cosets < 2^32")
    fixed = _relation_relators(relations, 2 * generators)
    sub = [] if subgroup is None else _words(subgroup, 2 * generators)
    return fixed, sub


def run(op: str, family: Family, reduction: str = "all", prefix: int | None = None, **args):
    op = op.removeprefix("coset_enumeration.")
    generators = int(args["generators"])
    bound = int(args["max_cosets"])
    subgroup_arg = args.get("subgroup") if op != "is_finite" else None
    fixed, subgroup = _setup(family, generators, args["relations"], bound, subgroup_arg)
    members = list(itertools.islice(rt.iter_members(family), prefix))
    answers = []
    for member in members:
        word = list(map(int, member[0]))
        if any(letter >= 2 * generators for letter in word):
            raise ValueError("word contains a letter outside 0..2*generators-1")
        answers.append(Table(generators, bound).enumerate(fixed + [word], subgroup))

    if op == "index":
        return rt.reduce_int(reduction, [len(a[0]) if a is not None else 0 for a in answers], members, NATURALS)
    if op == "is_finite":
        return rt.reduce_bool(reduction, [a is not None for a in answers], members, NATURALS, **args)
    if op != "permutation_representation":
        raise ValueError(f"unknown operation {op}")
    if reduction != "all":
        raise ValueError("permutation representations only reduce with `all`")
    degrees = [len(a[0]) if a is not None else 0 for a in answers]
    images = [0] * (len(answers) * generators * bound)
    for i, answer in enumerate(answers):
        if answer is None:
            continue
        for g, permutation in enumerate(answer):
            base = (i * generators + g) * bound
            images[base:base + len(permutation)] = permutation
    return CosetRepresentations(len(answers), generators, bound, degrees, images)
