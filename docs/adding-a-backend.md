# Adding a backend

You are here because a module is correct and you want it faster on hardware we have. This is the
most valuable kind of change in the repository and also the most contained: a backend touches one
directory, registers itself, and is held to the same oracle as every other one. Nothing outside
the module can tell which backend answered.

## What a backend is

A backend is a source file (or a few) under `modules/<module>/backends/<name>/` that registers a
`Backend` with the runtime (`runtime/src/registry.hpp`):

- `available()`: can it run on this machine? A CPU-feature backend checks `cpu_has("avx512bw")`;
  a GPU backend checks for a device. The runtime asks once.
- `accepts(request)`: does it handle this particular request? Prime range, shape limits, which
  operations, which families. Be exact; the runtime believes you.
- `run(request)`: produce the result object.
- `specificity`: a number; among the accepting backends the runtime picks the highest. The
  portable one is 0. A backend that requires a CPU feature or a device should be above it.

The manifest entry (`modules/<module>/manifest.toml`, `[[backends]]`) names the source files and
any CPU features they need; `tools/manifest.py generate` turns that into build rules, so you do
not edit CMake. The `accepts` text there is documentation for callers; the C++ `accepts` is what
the runtime enforces. Keep them saying the same thing.

## The contract you are held to

Same bytes as the reference, for every request you accept. The tests state the kernel's answers
as Lean `example`s over the module's `Reference.lean` and Lean checks them with `decide`; there is
no expected-output file to update, and nothing to "adjust" when a case disagrees. If Lean rejects
a claim, the backend is wrong for that input (or `accepts` is too generous). Both are fixed in
the backend.

Canonical choices matter as much as values. Nullspace bases, rref, member order, hit indices: the
reference defines exactly one answer, and yours has to be that one. `all` reductions and `hits`
make the order visible, so they are the quickest way to find a canonicalisation slip.

## The gfp walk, for backends that use it

gfp's walk operations (rank, nullity, full_row_rank, full_col_rank, in_span, rref, nullspace)
share a driver, `modules/gfp/backends/walk.hpp`. It walks the family tree depth-first, keeps a
basis in step with the walk, prunes subtrees the reduction cannot see, splits the top-level
branches across threads, merges, and checks completeness. A backend that keeps that structure
supplies only a `Basis` type: how to reduce one row against the rows already held, how to undo
the last addition, how to keep a target reduced for in_span, and how to read the rref back out.
The interface is written at the top of `walk.hpp`; `generic/field.hpp` is the reference
implementation of it, and `generic/gfp_generic.cpp` shows the registration and the dispatch
between walk operations and the explicit-only ones.

Where the time goes in the generic basis, so you know what to attack:

- `add()` reduces one incoming row against up to `rank` held rows, `cols` entries each, with a
  Barrett reduction per entry. That inner loop is almost all of the runtime for small primes.
  Packing (bits for p = 2, u8/u16 lanes for small p) and SIMD are the obvious wins; the AVX-512BW
  machine this was written on has 32 u16 lanes per register.
- `rref()` runs only at leaves of `all` reductions of rref/nullspace; it is not on the hot path
  for counts and histograms.
- Threads split the top-level branches. Families whose top level is narrow (a Grassmannian with
  few pivot sets) leave cores idle; a backend can split deeper if it wants, as long as indices
  stay canonical.

A backend need not use the walk at all. A GPU backend will more likely unrank members in bulk
from their canonical indices (`Family::member` is the scalar version of that) and reduce on
device. The only things fixed are the request in, the object out, and the bytes matching.

## Doing it

1. Read `PHILOSOPHY.md`, the module's `manifest.toml`, and its `Reference.lean`. Ten minutes.
2. Copy the generic backend's shape into `backends/<name>/`, change what you are changing, add
   the `[[backends]]` entry, run `tools/manifest.py generate`, rebuild.
3. `pytest -n auto tests`. The tests parametrise over every available backend, so
   yours is covered the moment `describe()` lists it. Force it in a session with
   `lk.Context("<module>.<name>")` when you want to poke at it by hand.
4. `tools/bench.py --module <module> --backend <module>.<name>` for the numbers. Add a case with
   `bench=` to `cases.py` if your backend's sweet spot is not represented; the bench asserts kernel and naive
   agree, so a new case is also a test.
5. Commit with the numbers in the message, before and after.

## When you are stuck

If the bytes will not match and you cannot see why, shrink the input until they do and diff the
first member that differs; `ctx.member(family, i)` and `naive.py` give you the member and the
slow answer for it. If the reference itself looks wrong to you, say so in the commit or the
thread rather than making the backend match a definition you think is mistaken; the reference can
be changed, but it is changed on purpose and in one place.

If the hardware you want is not on this machine, the backend cannot be validated here and should
not be merged as if it were. Say what it needs.
