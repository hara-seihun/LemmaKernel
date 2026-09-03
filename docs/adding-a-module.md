# Adding a module

A module is the unit of meaning in LemmaKernel: a set of object kinds, families, operations, and
reductions whose inputs and outputs are stable, with everything beneath them replaceable. Before
adding one, make sure the thing you want is not an operation on an existing module; a new
operation is a much smaller change than a new module.

## What a module owns

`modules/<name>/`, containing:

- `manifest.toml`: the one declaration of the module. Kinds, families, reductions, operations,
  backends, and the paths below. `tools/manifest.py generate` derives the C++ registry data, the
  runtime `describe()` JSON, the CMake source list, the Lake libraries, and the Python-side
  manifest from it; `tools/manifest.py check` (run at CMake configure) refuses stale output.
- `lean/<Name>.lean`, `lean/<Name>/Reference.lean`, `lean/<Name>/Contract.lean`. The reference
  is the executable definition of every operation, family order, and reduction, written as
  structural recursion so `decide +kernel` can evaluate it; it is the oracle for every test. The
  contract states, against Mathlib, what the reference means. Unproved statements carry `sorry`
  and say so; that is honest and expected.
- `backends/`: at least a portable one. See [adding-a-backend.md](adding-a-backend.md).
- `naive/naive.py`: the obvious Python implementation. It is the benchmark baseline, and the
  tests hold it to the same oracle as the kernel, so it is also a readable second opinion.
- `tests/`: builds inputs, runs every available backend, states each answer as a Lean claim via
  `tools/leancheck.py`, and lets Lean decide. No expected answers in the file.
- `bench/bench.py`: kernel against naive on requests that look like real use, with speed-ups.

gfp is the worked example of all of this; copy its shape.

## Runtime pieces you may need to extend

The runtime (`runtime/src/`) currently knows one member type, matrices over F_p, and six family
kinds. A module over a different kind of object (permutations, graphs, polynomials) needs:

- an object kind in `object.hpp` with its interchange encoding in `object.cpp` (header plus flat
  little-endian arrays; see `docs/interchange.md`);
- families over that object in `family.hpp`/`family.cpp`, with a canonical order, a closed-form
  size, `member(index)` for unranking, and a depth-first `enumerate` if backends want to share
  prefix work;
- the matching `lk_family_*` constructors in `lk.h`, and their Python wrappers.

Keep the C ABI additive: new functions, never changed signatures.

## The reference, and `decide`

Write `Reference.lean` for the kernel evaluator, not for elegance:

- structural recursion only (a `fuel` argument is fine; well-founded recursion does not reduce);
- `Nat` with `% p` everywhere; the kernel accelerates `Nat` arithmetic, comparison and `pow`,
  but `pow` computes the whole power, so modular exponentiation must be written by hand
  (`Gfp.powMod`);
- `List`, `Option`, tuples, and inductives with `deriving DecidableEq`;
- one `run` function from (operation, family, reduction) to a `Result` inductive, returning
  `.invalid` for combinations the manifest does not allow, so tests can check that the runtime's
  rejections and the reference's agree.

Budget: the kernel evaluates roughly a hundred small matrix eliminations per second. Test
families of a few dozen members cost a few seconds per claim file; keep it there. Larger checks
belong in the bench, where kernel and naive are compared byte for byte.

## Doing it

1. Write the manifest and the reference first, together. If an output has a choice in it (a
   basis, an order, a representative), the reference makes the choice and the manifest's
   `summary` says what it is. Until you can state the canonical answer, the operation is not
   ready.
2. Write the naive implementation and the tests, and make the naive implementation pass the
   oracle. Now you have a correct, slow module.
3. Write the portable backend. Make it pass. Bench it.
4. Write the contract. Statements first; proofs when a small primitive admits one.
5. Add the module to the table in `README.md`, run `tools/manifest.py generate`, commit.

## When you are stuck

If the reference cannot be made to evaluate in reasonable time, that is worth saying out loud
rather than shrinking tests until they say nothing; the reference can be restructured (chunked
lists, fewer recomputations) and the harness can grow. If the operation you want has no unique
answer, do not paper over it with "equivalent up to"; find the canonical form or report that the
operation as specified is not a function.
