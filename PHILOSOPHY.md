# Philosophy

LemmaKernel is a compute platform for mathematics whose users are agents. This file records the
decisions that shape every module, and why. Read it before adding or changing one.

## Who this is for

Agents, exploring. The typical session is an agent testing a conjecture by computing something at
a size no one has computed before, hundreds of times in an afternoon. Speed at that moment decides
which conjectures get tested at all. Developer comforts do not matter; a function that takes some
input and returns some output, unbelievably fast, is the whole product.

The CI-classification campaign that motivated this (`/home/kenan/work/ci-classification`) spent
three weeks with 7-minute Python scripts, a 29-minute census, and boards that "exceeded the time
budget". Its open rows are open partly because nobody could compute at the next size. That is the
failure this platform exists to remove.

## Modules are the only stable surface

A module is a named unit with a manifest, a Lean contract, a naive reference implementation, one
or more backends, tests, and a benchmark. The module's inputs, outputs, and their meaning are
stable. Everything beneath is disposable: backends may change layout, algorithm, or device
between versions, and nothing outside the module may depend on how they work.

This is deliberate. The fastest algorithm for a query is rarely the first one written, and we want
to be free to replace it wholesale without touching a caller.

## Inputs are descriptions, not data

Nobody who calls this platform has a million explicit matrices. They have "every 4-dimensional
subspace of this 6-dimensional space, mapped through this fixed matrix, with this row appended",
and they want to know whether any of them satisfies a predicate. Handing over a million matrices
means the caller did the enumeration in Python and paid to move the result across the boundary.

So the unit of input is a **family**, a small description of a set of mathematical objects with a
canonical enumeration order, plus a **query** (what to compute per member) and a **reduction**
(what to return: a count, a histogram, the hits, or everything). The kernel enumerates internally,
shares work between members that share structure, prunes subtrees the reduction cannot see, and
returns only the answer.

Families are a platform concept. Each module contributes the member types and queries it
understands; the enumeration framework is shared.

Explicit data is the degenerate family with one member per item. It still works; it is just not
the design centre.

## Objects cross the boundary as representations

Every object kind has exactly one interchange encoding: a small header and flat little-endian
arrays, deliberately dumb. It is what a caller imports and exports, what the Lean side parses, what
tests compare, and what gets written to disk. Inside the library, an object is an opaque handle and
the backend keeps whatever layout it likes, on whatever device it likes. Composing operations never
leaves the library's memory.

## Every answer is bit-identical across backends

For any input, the exported bytes of a result are the same whichever backend produced it. This
forces every operation to have a unique answer. Where mathematics leaves a choice (a basis of a
nullspace, the order of an enumeration), the contract picks one and every backend canonicalises to
it. "Equivalent" is never accepted.

This is also what makes automatic backend selection safe: the caller can never observe which one
ran.

## Every module has a Lean contract

The contract states, against Mathlib's definitions, what each output means. It is not a proof that
the native code is correct; there is no tooling for that and we do not pretend. It is the exact
statement of what a caller is assuming when they rely on the module, written once, in the language
a theorem-side reader needs.

A test enforces that every module has one. Consistency is worth more than deciding case by case
which modules "need" it.

Alongside the contract, each module has an executable reference in Lean of the same operations.
It is the oracle: tests state the kernel's answers as Lean examples and `decide` evaluates the
reference to accept or reject them. No test stores an expected answer; the only place an answer
is defined is the reference. For small primitives that reference can be proved to match the
contract; for large ones it is at least stated against Mathlib.

## Existence has a witness, non-existence has a count

An operation that answers "does something exist?" returns the thing when it exists, in a form a
small checker can verify without redoing the search. An operation that answers "nothing exists"
cannot do that, so it returns the number of candidates it examined next to the closed-form size of
the family, and refuses to report if they differ. That is the practical substitute for proving an
enumeration exhaustive, and it is part of every non-existence answer.

## Every module has a naive implementation

The naive implementation does the same job the obvious way: materialise every member, compute from
scratch. It is the benchmark baseline, so benchmarks report both numbers, and it is a second
opinion a reader can follow by eye. It is held to the same Lean oracle as the kernel; it is not
the oracle.

## One source of truth

A module's manifest is the only place its object kinds, families, operations, reductions,
backends, and contract paths are declared. Everything else (the C-side registry, the description a
caller can query at runtime, the Python wrappers, the checks) is generated from it. If something
about a module is written down twice, one of the copies is wrong.

## Backends declare what they accept

The contract covers the whole mathematical domain (any prime, any size). A backend declares the
subset it handles, and the runtime picks the most specific backend that accepts a request. Not
every operation has every backend; we write backends for hardware we have and let others contribute
for hardware they have.

## What we do not do

- No security controls between backends, modules, or callers. One trusted machine, one trusted
  agent population.
- No compatibility layers, holding areas, or deprecated paths. Replace, migrate consumers, delete.
- No fallbacks. If the selected backend cannot run a request, the call fails and says why.
