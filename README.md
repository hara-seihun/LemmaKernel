# LemmaKernel

A compute platform for mathematics whose users are agents. It provides functions that take a
description of a set of mathematical objects, compute something about every member, and return
the answer, fast. [PHILOSOPHY.md](PHILOSOPHY.md) explains the decisions behind it; read that
first if you are going to change anything.

## Use it

```python
import lemmakernel as lk
ctx = lk.Context()                                  # every backend, every core
G = ctx.grassmannian(p=2, n=10, h=4)                # all 4-dim subspaces of F_2^10 (53 million)
F = ctx.stack(G, lk.matrix(2, [[1,1,0,0,1,0,1,1,0,1], [0,1,1,1,0,0,0,1,1,0]]))
hits = ctx.value("gfp.in_span", F, "hits", target=lk.matrix(2, [[1,0,1,1,0,0,1,0,1,1]]), limit=4)
hits.total, hits.family_size, hits.members.tolist()  # how many, out of how many, and four witnesses

D24 = ctx.perms(24, [[(i + 1) % 24 for i in range(24)], [(-i) % 24 for i in range(24)]])
beads = ctx.subsets(lk.matrix(2, [[int(i == j) for j in range(24)] for i in range(24)]), 8)
ctx.value("orbits.is_canonical", beads, "count", group=D24).value   # 15581 bracelets, 8 black of 24 beads
```

`lk.describe()` lists every module compiled in: its object kinds, families, operations,
reductions, backends, and which backends run on this machine. The Python binding is a thin
`ctypes` layer over the C ABI in [`runtime/include/lemmakernel/lk.h`](runtime/include/lemmakernel/lk.h);
anything Python can do, C can do with the same names.

Vocabulary, in the order a request is built:

- **family**: a description of a set of objects with a canonical order (`explicit`, `subsets`
  of a dictionary, `subsets_of` another family, `grassmannian`, `all_matrices`,
  `symmetric_matrices`, the `transform`/`stack` wrappers, `group_elements` of a permutation
  group, `group_tables` from stored Cayley tables or permutation generators, `range`/`words` of
  natural numbers, constrained `partitions`, `compositions`, and `standard_tableaux`). Family sizes in the hundreds of
  millions are normal.
- **operation**: what to compute per member (`gfp.rank`, `gfp.in_span`, `gfp.rref`, ...).
- **reduction**: what to bring back. For booleans: `count`, `hits`, `first` (the least hit,
  stopping early), `all`. For integers: `histogram`, `sum`, `max`, `min` (with the member
  attaining it), `all`.

Every reduced result carries `visited` and `family_size`; they are equal (or, for `first`,
`visited` is the hit's index + 1) or the call fails. That is the completeness check for a
non-existence answer.

## Build and test

```
cmake -S . -B build -G Ninja && ninja -C build      # liblemmakernel.so
pytest -n auto tests                                # every module's cases against every backend, naive, and Lean
tools/bench.py                                      # kernel vs naive on each module's bench cases
```

The Lean side (`lake build`) needs Mathlib. In the canonical checkout `.lake/packages` is a
hardlink copy of `~/projects/LemmaLib/.lake/packages`, which pins the same Lean and Mathlib
versions. A task checkout (under `~/work/clones`) should share it rather than fetch its own:

```
mkdir -p .lake && ln -s /home/kenan/projects/LemmaKernel/.lake/packages .lake/packages
```

after which `lake build` finishes in seconds. Tests only need `Lk.Reference` and the module's
`Reference` (no Mathlib import) and take a few seconds each; on a shared machine use
`pytest -n 8 tests` rather than `auto`.
No test or bench script is written per module: a module ships `cases.py` and the manifest, and
`tools/harness.py` derives the rest.

## Layout

```
runtime/        the C ABI, object encoding, family enumeration, backend registry, reductions;
                runtime/lean/Lk is the Lean reference and contract for families and reductions
modules/NAME/   one module: manifest, Lean contract and reference, backends, naive, cases
python/         the binding; python/lemmakernel/_manifest.py is generated
tools/          manifest.py (generator and checker), harness.py (runs cases), bench.py, leancheck.py
tests/          the pytest entry points: every module's cases, runtime checks, manifest and Lean build
docs/           how to add a module or a backend
```

Every derived file is produced by `tools/manifest.py generate` from `modules/*/manifest.toml`
and committed; the build refuses to configure if they are stale.

## Modules

| module | what it computes | backends |
|---|---|---|
| [automorphisms](modules/automorphisms/manifest.toml) | automorphism groups of finite groups: order, canonical generators, holomorph order, and outer automorphism group order | `generic` (propagated partial table isomorphisms, portable C++) |
| [burnside](modules/burnside/manifest.toml) | orbit counts, fixed counts, and canonical cycle indices for permutation groups on subsets and words, without enumerating the acted-on family | `generic` (cycle-type formulas, portable C++) |
| [cayley](modules/cayley/manifest.toml) | simple undirected Cayley graphs of finite permutation groups: connectivity, regularity, girth, diameter, graph automorphism order, and the CI property of a connection set | `generic` (exact group closure, graph search, partition refinement and individualisation, portable C++) |
| [designs](modules/designs/manifest.toml) | whole-family tests and data for finite block designs: t-design multiplicities, resolutions, dual 2-designs, intersections, and Kramer-Mesner matrices | `generic` (portable C++) |
| [difference_sets](modules/difference_sets/manifest.toml) | difference sets, difference multisets, regular partial difference sets, and relative difference sets in finite permutation groups | `generic` (depth-first multiplicity pruning, portable C++) |
| [elliptic_curves_fp](modules/elliptic_curves_fp/manifest.toml) | curves `y^2 = x^3 + ax + b` over `F_p` (`p > 3`) from families of `(a, b)` pairs: point counts, singularity, supersingularity, j-invariants, `F_p`-isomorphism classes and the invariant factors of `E(F_p)` | `generic` (per-request square-root tables, per-member third differences, portable C++) |
| [gfp](modules/gfp/manifest.toml) | linear algebra over F_p on families of matrices: rank, rref, nullspace, span membership, solve, inverse, rref witness | `generic` (any p < 2^32, portable C++) |
| [gfq](modules/gfq/manifest.toml) | linear algebra over explicitly presented GF(q): rank, rref, nullspace, span membership, solve, inverse | `generic` (polynomial-basis arithmetic for q < 2^32, portable C++) |
| [graph_iso](modules/graph_iso/manifest.toml) | canonical forms, canonical vertex orders, and complete automorphism groups of undirected F_2 adjacency matrices, with loops allowed | `generic` (Weisfeiler-Lehman refinement and individualisation, portable C++) |
| [hadamard](modules/hadamard/manifest.toml) | Hadamard, skew, regular and conference predicates for F_2 matrices read as signs, plus signed-equivalence canonical forms | `generic` (portable C++) |
| [hypergraphs](modules/hypergraphs/manifest.toml) | uniform hypergraphs: linearity, weak colouring number, Berge cycles and girth, finite Turan searches, and two-colour Ramsey searches | `generic` (bit sets and backtracking, portable C++) |
| [integer_partitions](modules/integer_partitions/manifest.toml) | constrained partitions and compositions: number of parts, largest part, and rank/crank distributions | `generic` (portable C++ enumeration) |
| [lattice_of_subspaces](modules/lattice_of_subspaces/manifest.toml) | Gaussian binomials, flag counts, counts above and below a row space, and subspace incidence predicates | `generic` (checked formulas and row reduction, portable C++) |
| [matrix_groups](modules/matrix_groups/manifest.toml) | order, irreducibility, absolute irreducibility, invariant bilinear forms, and imprimitivity for groups generated by subsets of a matrix dictionary | `generic` (Schreier stabilizer chain and finite-field searches, portable C++) |
| [matchings_and_flows](modules/matchings_and_flows/manifest.toml) | weighted bipartite perfect-matching counts, weighted undirected spanning-tree counts, and directed maximum-flow values from integer matrices | `generic` (Ryser permanent, matrix-tree determinant, and Edmonds-Karp, portable C++) |
| [orbits](modules/orbits/manifest.toml) | finite groups acting on families: orbit representatives (`is_canonical`), canonical index, orbit size, stabiliser order, fixed points for Burnside, and the projective action that turns a matrix group into permutations of points | `generic` (per-member orbit search, portable C++) |
| [perm_groups](modules/perm_groups/manifest.toml) | Schreier-Sims for permutation groups: order, membership, transitivity, primitivity, point-orbit partitions, and deterministic bases with strong generators | `generic` (portable C++, parallel across groups) |
| [permutation_statistics](modules/permutation_statistics/manifest.toml) | inversions, descents, major index, dense complete-cycle-type codes, classical pattern avoidance, and strong Bruhat order on permutation families | `generic` (parallel scans, pruned pattern matching, portable C++) |
| [polynomials_fq](modules/polynomials_fq/manifest.toml) | univariate polynomials over `F_q` on families of coefficient rows: irreducibility, factorisation degrees, primitivity, the order of `x`, roots and gcd | `generic` (Frobenius distinct-degree factorisation, portable C++) |
| [projective_sets](modules/projective_sets/manifest.toml) | arcs, caps, blocking sets, hyperovals, ovoids, span rank, maximum collinearity, and secant, tangent and passant counts for point sets in `PG(n,p)` | `generic` (precomputed line and hyperplane incidence, portable C++) |
| [residues](modules/residues/manifest.toml) | arithmetic in Z/n over ranges of residues: multiplicative order, primitive roots, quadratic residues, discrete logarithms by baby-step giant-step, the Legendre and Jacobi symbols, and the least primitive root of each modulus in a range | `generic` (factorisation of the modulus shared by every member, portable C++) |
| [set_systems](modules/set_systems/manifest.toml) | extremal predicates on finite set systems: intersecting families, antichains, sunflowers, maximum degree, lower shadows, and exact EKR/Sperner extremality | `generic` (prefix-pruned enumeration, portable C++) |
| [simplicial_complexes](modules/simplicial_complexes/manifest.toml) | abstract simplicial complexes read off 0/1 matrices, either generated by the rows or as the independence complex of the forbidden rows: f-vector, total faces, Euler characteristic, Betti numbers over F_p, and Bjorner-Wachs shellability | `generic` (faces as 64-bit masks, boundary elimination as bitsets over F_2 and dense over F_p, shelling search over facet subsets, portable C++) |
| [subspace_orbits](modules/subspace_orbits/manifest.toml) | GL, PGL and PGammaL orbits of row spaces from Grassmannians and their transform/stack derivatives; canonical indices use the least rref in Grassmannian order | `generic` (per-member rref orbit search, portable C++) |
| [young](modules/young/manifest.toml) | partitions and standard Young tableaux, hook-length counts, Kostka numbers, RSK, and irreducible S_n character values by Murnaghan-Nakayama | `generic` (portable C++) |

`ctx.group_tables(lk.naturals(tables))` builds a family from stored Cayley tables;
`ctx.generated_group(permutations)` builds one from permutation generators.
`automorphisms.aut_generators` scans all automorphisms in lexicographic order and keeps an
automorphism exactly when earlier choices do not generate it. This fixes one generating list for
every Cayley-table labelling.

A `designs` block on `v` labelled points is a `k x v` matrix of distinct standard basis rows in
increasing point order. The family members are the indexed blocks. Kramer-Mesner rows and columns
use the least lexicographic member of each subset orbit, so their matrix is canonical.

For `gfq`, tag matrices and families with `q`. Pass the monic irreducible polynomial as
`modulus=[c_0, ..., c_(d-1), 1]` in increasing-degree order over F_p. The integer label of a field
element is its little-endian base-p coefficient vector. For example, GF(4) uses `q=4` and may use
`modulus=lk.matrix(4, [[1, 1, 1]])` for x^2 + x + 1.

An orbit is a permutation of member indices; the representative is the least index. Permutation
groups act on `subsets` families, matrix groups on `grassmannian` and `all_matrices` families
(on the right, `M ↦ M A`). `tools/bench.py` compares each module's generic backend with the
naive Python implementation on inputs small enough for naive to finish, so every row is a
byte-for-byte agreement; single-threaded ratios on this machine run from about 40× (Grassmannian
orbits, where each step is an elimination) to a few thousand× (independence of subsets;
isomorphism classes of elliptic curves), and higher where the backend changes the algorithm
(Burnside counts from cycle types).

`elliptic_curves_fp` reads a `1 x 2` member as the pair `(a, b)` of `y^2 = x^3 + ax + b`, so
`all_matrices(p, 1, 2)` is every curve over `F_p` and a histogram of `point_count` over it is the
isogeny classes; families of another shape are refused rather than reinterpreted.

`residues` reads a `range` family as residues and takes the modulus as an argument, except
`least_primitive_root`, whose member is the modulus. Where mathematics leaves a choice it makes
one: an order is 0 for a non-unit, a discrete logarithm is the least `x < n` and the modulus
itself when there is none, a quadratic symbol is 0, 1 or 2 with 2 standing for -1, and the least
primitive root is 0 when the unit group is not cyclic.

`polynomials_fq` reads a `1 x d` member `[a_0, ..., a_{d-1}]` as the monic polynomial
`x^d + a_{d-1} x^(d-1) + ... + a_0`: entry `i` is the coefficient of `x^i` and the degree is the
number of columns, so `all_matrices(q, 1, d)` is every monic polynomial of degree `d` and
`transform`, `subsets` and `explicit` give the derived families. `order` is Lidl and
Niederreiter's: for `f = x^h g` with `g(0)` nonzero it is the least `e` with `g` dividing
`x^e - 1`.

`subspace_orbits` treats each matrix as a row space, so it also handles transforms and stacks that
change the displayed basis or its rank. Set `projective=0` to retain the generated GL group order,
or `projective=1` to identify nonzero scalar matrices. The runtime currently works over prime
fields, where PGammaL has the same action as PGL.

To add to this table, read [docs/adding-a-module.md](docs/adding-a-module.md). To make an
existing module faster, read [docs/adding-a-backend.md](docs/adding-a-backend.md).

## Where it lives

Source: `~/projects/LemmaKernel`, public remote `hara-seihun/LemmaKernel`. It is not yet
deployed anywhere outside the checkout; the shared-library and `/srv/pi` publication step is open.
