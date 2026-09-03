"""Direct subgroup enumeration for finite permutation groups.

Every subgroup is stored as the sorted list of all its elements. Starting with the identity,
the implementation adjoins each parent element and closes the result, deduplicating complete
subgroups. This is intentionally straightforward. The C++ backend uses the same mathematics with
an indexed multiplication table.
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "python"))
from lemmakernel import naive as rt  # noqa: E402
from lemmakernel.interchange import Family, Perms, SubgroupLists  # noqa: E402


def compose(a, b):
    return tuple(b[a[i]] for i in range(len(a)))


def inverse(g):
    out = [0] * len(g)
    for i, x in enumerate(g):
        out[x] = i
    return tuple(out)


def closure(n, generators):
    identity = tuple(range(n))
    generators = [tuple(g) for g in generators]
    seen = {identity}
    queue = [identity]
    for a in queue:
        for g in generators:
            b = compose(a, g)
            if b not in seen:
                seen.add(b)
                queue.append(b)
    return sorted(seen)


def valid_generators(n, generators):
    target = list(range(n))
    return n > 0 and all(len(g) == n and sorted(g) == target for g in generators)


class Enumerator:
    def __init__(self, n, generators):
        self.n = n
        self.elements = closure(n, generators)
        self.index = {g: i for i, g in enumerate(self.elements)}
        self.table = [[self.index[compose(a, b)] for b in self.elements] for a in self.elements]
        self.inverses = [self.index[inverse(g)] for g in self.elements]
        self._subgroups = None

    def generated_indices(self, generators):
        seen = {0}
        queue = [0]
        for a in queue:
            for g in generators:
                b = self.table[a][g]
                if b not in seen:
                    seen.add(b)
                    queue.append(b)
        return tuple(sorted(seen))

    def join(self, subgroup, element):
        return self.generated_indices((*subgroup, element))

    def subgroups(self):
        if self._subgroups is not None:
            return self._subgroups
        found = {(0,)}
        queue = [(0,)]
        for subgroup in queue:
            inside = set(subgroup)
            for element in range(len(self.elements)):
                if element in inside:
                    continue
                extension = self.join(subgroup, element)
                if extension not in found:
                    found.add(extension)
                    queue.append(extension)
        self._subgroups = sorted(found)
        return self._subgroups

    def conjugate(self, subgroup, g):
        inverse_g = self.inverses[g]
        return tuple(sorted(self.table[self.table[inverse_g][h]][g] for h in subgroup))

    def canonical_conjugate(self, subgroup):
        return min(self.conjugate(subgroup, g) for g in range(len(self.elements)))

    def conjugacy_classes(self):
        return [h for h in self.subgroups() if self.canonical_conjugate(h) == h]

    @staticmethod
    def contained(a, b):
        return set(a) <= set(b)

    def maximal_subgroups(self):
        subgroups = self.subgroups()
        whole = len(self.elements)
        maximal = []
        for h in subgroups:
            if len(h) == whole:
                continue
            if any(len(h) < len(k) < whole and self.contained(h, k) for k in subgroups):
                continue
            if self.canonical_conjugate(h) == h:
                maximal.append(h)
        return maximal


def normal_in(n, parent_generators, candidate_generators):
    parent = closure(n, parent_generators)
    candidate = closure(n, candidate_generators)
    parent_set = set(parent)
    candidate_set = set(candidate)
    if not candidate_set <= parent_set:
        return False
    for g in parent:
        gi = inverse(g)
        for h in candidate:
            if compose(compose(gi, h), g) not in candidate_set:
                return False
    return True


def subgroup_lists(per_group):
    group_offsets = [0]
    subgroup_offsets = [0]
    elements = []
    for groups in per_group:
        group_offsets.append(group_offsets[-1] + len(groups))
        for subgroup in groups:
            elements.extend(subgroup)
            subgroup_offsets.append(len(elements))
    return SubgroupLists(len(per_group), group_offsets, subgroup_offsets, elements)


def run(op: str, family: Family, reduction: str = "all", prefix: int | None = None, **args):
    op = op.removeprefix("subgroups.")
    if family.kind not in ("subsets", "explicit"):
        raise ValueError(f"{op} needs subsets or explicit families of permutation rows")
    (dictionary,) = family.children
    if not isinstance(dictionary, Perms):
        raise ValueError("family members must contain permutation rows")
    n = dictionary.n
    groups = list(itertools.islice(rt.iter_members(family), prefix))
    if any(not valid_generators(n, generators) for generators in groups):
        raise ValueError("family members must contain permutation rows of one degree")

    if op == "is_normal":
        parent = args.get("group")
        if not isinstance(parent, Perms):
            raise ValueError("is_normal needs a permutation group argument")
        if parent.n != n:
            raise ValueError("candidate and parent must have the same degree")
        parent_generators = parent.tolist()
        if not valid_generators(n, parent_generators):
            raise ValueError("group must contain permutation rows")
        flags = [normal_in(n, parent_generators, candidate) for candidate in groups]
        return rt.reduce_bool(reduction, flags, groups, 0, **args)

    enumerators = [Enumerator(n, generators) for generators in groups]
    if op == "subgroup_count":
        values = [len(e.subgroups()) for e in enumerators]
        return rt.reduce_int(reduction, values, groups, 0)
    if op == "conjugacy_classes":
        return rt.reduce_values(reduction, subgroup_lists([e.conjugacy_classes() for e in enumerators]))
    if op == "maximal_subgroups":
        return rt.reduce_values(reduction, subgroup_lists([e.maximal_subgroups() for e in enumerators]))
    raise ValueError(f"unknown operation {op}")
