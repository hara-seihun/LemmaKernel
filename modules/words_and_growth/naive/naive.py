"""Direct word enumeration for finitely presented group growth."""
from __future__ import annotations

import itertools
import sys
from functools import cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "python"))
from lemmakernel import naive as rt  # noqa: E402
from lemmakernel.interchange import Family, Matrix, NATURALS  # noqa: E402

MAX_NORMALISE_STEPS = 1 << 20
MAX_CRITICAL_PAIRS = 1 << 20


def shortlex_less(left, right):
    return (len(left), left) < (len(right), right)


def replace_at(word, position, length, replacement):
    return word[:position] + replacement + word[position + length:]


def rewrite_once(rules, word):
    for position in range(len(word) + 1):
        for left, right in rules:
            if word[position:position + len(left)] == left:
                return replace_at(word, position, len(left), right)
    return None


def normalise(rules, word):
    word = tuple(word)
    for _ in range(MAX_NORMALISE_STEPS):
        following = rewrite_once(rules, word)
        if following is None:
            return word
        word = following
    raise ValueError("normal-form computation exceeded the step limit")


def critical_pairs(rules):
    checked = 0
    for outer_left, outer_right in rules:
        for inner_left, inner_right in rules:
            if len(inner_left) <= len(outer_left):
                for position in range(len(outer_left) - len(inner_left) + 1):
                    if outer_left[position:position + len(inner_left)] != inner_left:
                        continue
                    checked += 1
                    if checked > MAX_CRITICAL_PAIRS:
                        raise ValueError("rewriting system has too many critical pairs")
                    yield outer_right, replace_at(outer_left, position, len(inner_left), inner_right)
            for overlap in range(1, min(len(outer_left), len(inner_left)) + 1):
                if outer_left[-overlap:] != inner_left[:overlap]:
                    continue
                checked += 1
                if checked > MAX_CRITICAL_PAIRS:
                    raise ValueError("rewriting system has too many critical pairs")
                yield outer_right + inner_left[overlap:], outer_left[:-overlap] + inner_right


def build_rules(generators: int, relations: Matrix):
    if not 1 <= generators <= 128:
        raise ValueError("generators must be between 1 and 128")
    if not isinstance(relations, Matrix) or relations.p != NATURALS:
        raise ValueError("relations must be an lk.naturals matrix of encoded equations")
    alphabet = 2 * generators
    rows = rt.vectors_of(relations)
    if len(rows) > 256 or relations.cols > 514:
        raise ValueError("presentation exceeds the equation count or width limit")
    rules = []

    def add_rule(left, right):
        left, right = tuple(left), tuple(right)
        if left == right:
            return
        rule = (right, left) if shortlex_less(left, right) else (left, right)
        if rule not in rules:
            rules.append(rule)

    for row in rows:
        if len(row) < 2:
            raise ValueError("relation row needs two lengths")
        left_length, right_length = row[:2]
        if left_length + right_length > len(row) - 2:
            raise ValueError("relation lengths exceed the row width")
        left = row[2:2 + left_length]
        right = row[2 + left_length:2 + left_length + right_length]
        if any(letter >= alphabet for letter in left + right):
            raise ValueError("relation has a symbol outside the generator alphabet")
        if any(letter != alphabet for letter in row[2 + left_length + right_length:]):
            raise ValueError("relation padding must equal the alphabet size")
        add_rule(left, right)
    for i in range(generators):
        add_rule((2 * i, 2 * i + 1), ())
        add_rule((2 * i + 1, 2 * i), ())
    for left, right in critical_pairs(rules):
        if normalise(rules, left) != normalise(rules, right):
            raise ValueError("rewriting system is not confluent: a critical pair does not join")
    return alphabet, rules


def words(alphabet, length):
    return itertools.product(range(alphabet), repeat=length)


def run(op: str, family: Family, reduction: str = "all", prefix: int | None = None, **args):
    op = op.removeprefix("words_and_growth.")
    alphabet, rules = build_rules(args["generators"], args["relations"])
    members = list(itertools.islice(rt.iter_members(family), prefix))

    @cache
    def sphere_size(radius):
        return sum(rewrite_once(rules, word) is None for word in words(alphabet, radius))

    @cache
    def ball_size(radius):
        return sum(sphere_size(length) for length in range(radius + 1))

    @cache
    def geodesic_count(radius):
        return sum(len(normalise(rules, word)) == radius for word in words(alphabet, radius))

    if op == "is_geodesic":
        if family.kind != "words":
            raise ValueError("is_geodesic is defined on words families only")
        if family.params["alphabet"] != alphabet:
            raise ValueError("words family alphabet must equal 2 * generators")
        flags = [len(normalise(rules, member[0])) == len(member[0]) for member in members]
        return rt.reduce_bool(reduction, flags, members, NATURALS, **args)

    if family.kind != "range":
        raise ValueError(f"{op} is defined on range families only")
    operation = {"ball_size": ball_size, "sphere_size": sphere_size, "geodesic_count": geodesic_count}.get(op)
    if operation is None:
        raise ValueError(f"unknown operation {op}")
    values = [operation(member[0][0]) for member in members]
    return rt.reduce_int(reduction, values, members, NATURALS)
