# Adding a module

A module is the unit of meaning in LemmaKernel: a set of object kinds, families, operations, and
reductions whose inputs and outputs are stable, with everything beneath them replaceable. Before
adding one, make sure the thing you want is not an operation on an existing module; a new
operation is a much smaller change than a new module.

## What a module owns

`modules/<name>/`, containing:

- `manifest.toml`: the one declaration of the module. Module-specific kinds (each with the Lean
  `Value` constructor that carries it), operations with typed arguments (`args = { target = "vector" }`;
  types are `int`, `vector`, `vectors`, `perms`, `group`, `family`), backends, `[[rejections]]`
  (requests the runtime must refuse, by case name and error text), and the paths below.
  Families, reductions, and shared object kinds such as `lk.naturals` are runtime-level and live
  in `runtime/manifest.toml`.
  `tools/manifest.py generate` derives the C++ registry data, the runtime `describe()` JSON, the
  CMake source list, the Lake libraries, and the Python-side manifest from it;
  `tools/manifest.py check` (run at CMake configure) refuses stale output.
- `lean/<Name>.lean`, `lean/<Name>/Reference.lean`, `lean/<Name>/Contract.lean`. The reference
  imports `Lk.Reference` (families, reductions, `Result α`) and defines the module's `Op`, its
  `Value` type, and `run : Op → Family → Red → Result Value`, as structural recursion so
  `decide +kernel` can evaluate it; it is the oracle for every test. The contract states, against
  Mathlib, what the reference means. Unproved statements carry `sorry` and say so; that is honest
  and expected.
- `backends/`: at least a portable one. See [adding-a-backend.md](adding-a-backend.md).
- `naive/naive.py`: the obvious Python implementation, `run(op, family, reduction, prefix=None,
  **args)`. It is the benchmark baseline, and the tests hold it to the same oracle as the
  kernel, so it is also a readable second opinion. `prefix` answers for the first members only;
  the bench uses it to sample. Families and reductions are not the module's business:
  `lemmakernel.naive` materialises any family (`members(desc)`) and reduces integer, boolean
  or materialised values (`reduce_int`, `reduce_bool`, `reduce_values`) exactly as the runtime
  and the Lean reference do, so a naive implementation is only its own mathematics per member.
- `cases.py`: `cases(ctx, rng)` returning `tools.harness.Case` objects (family, operation,
  arguments, which reductions, whether the Lean kernel can afford it, whether and how to bench
  it), and optionally `invariants(ctx)` for cross-operation identities on inputs beyond the
  oracle (Burnside for orbits, `T·A = R` for gfp witnesses).

There is no per-module test or bench script. `tests/test_cases.py` runs every case against every
backend, the naive implementation and the reference, checks every rejection, thread invariance
and roundtrips, and fails if some (operation, reduction) pair the manifest allows has no oracle
case. `tools/bench.py` times the cases that carry `bench`. Both are driven by the manifest
through `tools/harness.py`.

gfp is the worked example of all of this; copy its shape.

## Runtime pieces you may need to extend

The runtime (`runtime/src/`) knows labelled field matrices with entries below the field-size tag
(`gfp` requires a prime tag; `gfq` supplies an extension-field presentation), permutations (a
`Matrix` with p = 0), and natural-number matrices (p = `NATURALS`, kind `lk.naturals`, no
arithmetic meaning). It provides thirteen family kinds: `explicit`, `subsets` of a dictionary,
`subsets_of` another family, `grassmannian`,
`all_matrices`, `symmetric_matrices`, the `transform`/`stack` wrappers, `group_elements` of a
permutation group, `range`/`words` over naturals, and constrained `partitions`/`compositions`.
Look there before adding a family: k-subsets
of a group's elements is `subsets_of(group_elements(...))`, sign matrices are `all_matrices` over
F_2 read as signs, a v-set is `range(0, v)`. A module over a different kind of object (graphs,
polynomials) needs:

- an object kind in `object.hpp` with its interchange encoding in `object.cpp` (header plus flat
  little-endian arrays; see `docs/interchange.md`);
- families over that object in `family.hpp`/`family.cpp`, with a canonical order, a closed-form
  size, `member(index)` for unranking, `index_of` for ranking when a module will permute
  members (orbits does), and a depth-first `enumerate` if backends want to share prefix work;
- if the module's operations produce one integer or boolean per member, `runtime/src/reduce.hpp`
  already implements every reduction (`Shared`, `Accumulator`, `assemble`, `parallel_ranges`),
  including the early stop of `first`; gfp and orbits both use it. Ask `acc.exhausted(index)`
  before doing work for a member or subtree so that `first` can stop;
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

Budget: the kernel evaluates roughly a hundred small matrix eliminations per second, and a
million or so plain reduction steps. Oracle cases of a few dozen members cost a few seconds per
claim file; keep it there. Larger inputs go in cases with `oracle=False` (still checked for
thread invariance and, when benched, byte-for-byte against naive) and in `invariants`.

## Doing it

1. Write the manifest and the reference first, together. If an output has a choice in it (a
   basis, an order, a representative), the reference makes the choice and the manifest's
   `summary` says what it is. Until you can state the canonical answer, the operation is not
   ready.
2. Write the naive implementation and `cases.py`, and make the naive implementation pass the
   oracle (`pytest -n auto tests -k naive`). Now you have a correct, slow module.
3. Write the portable backend. Make it pass. `tools/bench.py --module <name>`.
4. Write the contract. Statements first; proofs when a small primitive admits one.
5. Add the module to the table in `README.md`, run `tools/manifest.py generate`, commit.

## When you are stuck

If the reference cannot be made to evaluate in reasonable time, that is worth saying out loud
rather than shrinking tests until they say nothing; the reference can be restructured (chunked
lists, fewer recomputations) and the harness can grow. If the operation you want has no unique
answer, do not paper over it with "equivalent up to"; find the canonical form or report that the
operation as specified is not a function.
