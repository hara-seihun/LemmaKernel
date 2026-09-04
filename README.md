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

window = ctx.range(1_000_000_000, 1_010_000_000)                    # ten million consecutive integers
ctx.value("sieve_ranges.is_prime", window, "count").value           # 482449 of them are prime
```

`lk.describe()` lists every module compiled in: its object kinds, families, operations,
reductions, backends, and which backends run on this machine. The Python binding is a thin
`ctypes` layer over the C ABI in [`runtime/include/lemmakernel/lk.h`](runtime/include/lemmakernel/lk.h);
anything Python can do, C can do with the same names.

Vocabulary, in the order a request is built:

- **family**: a description of a set of objects with a canonical order (`explicit`, `subsets`
  of a dictionary, `subsets_of` another family, `grassmannian`, `all_matrices`,
  `symmetric_matrices`, `alternating_matrices`, `sublattices` of a fixed integral lattice, the `transform`/`stack`
  wrappers, `group_elements` of a permutation or matrix group, `group_tables` from stored Cayley
  tables, permutation generators, or the catalogue of every group of an order up to 31, graph families (`all_graphs`, `edge_subgraphs`,
  `cayley_graphs`), `range`/`words` of natural numbers, `latin_squares`, constrained `partitions`,
  `compositions`, and `standard_tableaux`). Family sizes in the hundreds of millions are normal;
  sizes and member indices are 128-bit, so a pruned search can settle a family of 10^20 members.
- **operation**: what to compute per member (`gfp.rank`, `gfp.in_span`, `gfp.rref`, ...).
- **reduction**: what to bring back. For booleans: `count`, `hits`, `first` (the least hit,
  stopping early), `all`. For integers: `histogram` (one bin per value, so it is refused above
  2^22), `sum`, `max`, `min` (with the member attaining it), `all`.

Every reduced result carries `visited` and `family_size`; they are equal (or, for `first`,
`visited` is the hit's index + 1) or the call fails. That is the completeness check for a
non-existence answer.

## Build and test

```
cmake -S . -B build -G Ninja && ninja -C build      # liblemmakernel.so, -march=native; -DLEMMAKERNEL_MARCH=x86-64-v3 for a portable binary
pytest -n auto tests                                # every module's cases against every backend, naive, and Lean
tools/bench.py                                      # kernel vs naive; reruns only modules whose sources changed
```

The Lean side (`lake build`) needs Mathlib. In the canonical checkout `.lake/packages` is a
hardlink copy of `~/projects/LemmaLib/.lake/packages`, which pins the same Lean and Mathlib
versions. A task checkout under `~/work/clones` should make another hardlink copy rather than
fetch its own:

```
mkdir -p .lake && cp -al /home/kenan/projects/LemmaKernel/.lake/packages .lake/packages
```

Do not symlink this directory. `lake clean` follows package symlinks and deletes the shared build.
With the hardlink copy, cleanup removes only the checkout's directory entries.

Nothing then has to be fetched or elaborated from Mathlib, but each module's own `Reference` and
`Contract` still elaborate once per checkout, so the first `lake build` there costs several
minutes for all of them together; so does the first build after a change to
`runtime/lean/Lk/Reference.lean`, which every module imports. Under a short command timeout, build
one library at a time (`lake build Gfp`): a whole-project `lake build` starts more parallel work
than a short slice can finish, and killing it loses every job in flight.

Tests only need `Lk.Reference` and the module's `Reference` (no Mathlib import) and take a few
seconds each; on a shared machine use `pytest -n 8 tests` rather than `auto`.
No test or bench script is written per module: a module ships `cases.py` and the manifest, and
`tools/harness.py` derives the rest.

Bench results are committed: `modules/NAME/bench.json` holds the rows and a fingerprint of the
module tree, the module trees it includes, and the runtime, and [BENCHMARKS.md](BENCHMARKS.md)
is generated from every record. `tools/bench.py` reruns a module only when its fingerprint
changed (`--status` says which and why, `--force` reruns anyway), and the tests fail on a module
whose sources moved without a new record. A runtime change stales every record; rerun
`tools/bench.py` after one when you can, but it is not a test failure.

## Layout

```
runtime/        the C ABI, object encoding, family enumeration, backend registry, reductions;
                runtime/lean/Lk is the Lean reference and contract for families and reductions
modules/NAME/   one module: manifest, Lean contract and reference, backends, naive, cases
python/         the binding; python/lemmakernel/_manifest.py is generated
tools/          manifest.py (generator and checker), harness.py (runs cases), bench.py, leancheck.py,
                split_search.py (a long sum_free_and_additive search in resumable pieces),
                group_catalogue.py (enumerates the groups of small order in
                runtime/src/group_catalogue_data.cpp; run it only to change that catalogue)
tests/          the pytest entry points: every module's cases, runtime checks, manifest and Lean build
docs/           how to add a module or a backend
```

Every derived file is produced by `tools/manifest.py generate` from `modules/*/manifest.toml`
and committed; the build refuses to configure if they are stale.

## Modules

| module | what it computes | backends |
|---|---|---|
| [automorphisms](modules/automorphisms/manifest.toml) | automorphism groups of finite groups: order, canonical generators, holomorph order, and outer automorphism group order | `generic` (propagated partial table isomorphisms, portable C++) |
| [bilinear_invariants](modules/bilinear_invariants/manifest.toml) | rank, radical dimension, determinant square classes, nondegeneracy, and canonical congruence labels of symmetric and alternating forms over F_p | `generic` (symmetric Schur elimination and canonical-form assembly, portable C++) |
| [boolean_functions](modules/boolean_functions/manifest.toml) | nonlinearity, algebraic degree, Walsh spectra, bent and APN tests, and domain-affine canonical forms for Boolean truth tables | `generic` (fast Walsh and Mobius transforms, prefix-pruned affine canonicalisation, portable C++) |
| [burnside](modules/burnside/manifest.toml) | orbit counts, fixed counts, and canonical cycle indices for permutation groups on subsets and words, without enumerating the acted-on family | `generic` (cycle-type formulas, portable C++) |
| [cayley](modules/cayley/manifest.toml) | simple undirected Cayley graphs of finite permutation groups: connectivity, regularity, girth, diameter, graph automorphism order, and the CI property of a connection set | `generic` (exact group closure, graph search, partition refinement and individualisation, portable C++) |
| [cayley_iso](modules/cayley_iso/manifest.toml) | fixed-size inverse-closed Cayley connection sets: Aut(G)-class counts, unlabelled graph-isomorphism class counts, and CI or non-CI tests for group-table families | `generic` (shared exact table automorphisms and graph canonical forms, portable C++) |
| [char_poly](modules/char_poly/manifest.toml) | characteristic and minimal polynomials, rational canonical form, GL conjugacy labels, regularity, semisimplicity, and element order | `generic` (portable C++; bounded rational-form factor search and element orders) |
| [characters](modules/characters/manifest.toml) | exact ordinary character tables, induction, restriction, and Frobenius-Schur indicators for finite abelian permutation groups | `generic` (group closure and exact enumeration of the dual group, portable C++) |
| [circulants](modules/circulants/manifest.toml) | exact character spectra of circulant graphs and digraphs, isomorphism at corrected-Adam orders, canonical connection sets, and the cyclic CI/DCI classification | `generic` (unit multipliers and character sums, portable C++) |
| [code_equivalence](modules/code_equivalence/manifest.toml) | linear codes up to monomial equivalence: canonical form, canonical index, class representatives, orbit sizes, and monomial automorphism orders | `generic` (search over information sets rather than over the group, portable C++) |
| [continued_fractions_and_pell](modules/continued_fractions_and_pell/manifest.toml) | continued fractions of `sqrt(d)`, their period statistics, fundamental units of `Z[sqrt d]`, fundamental Pell solutions, solvability of the negative Pell equation, and class numbers of imaginary quadratic orders, over ranges of naturals | `generic` (one continued-fraction walk per member with 128-bit convergents, portable C++) |
| [coset_enumeration](modules/coset_enumeration/manifest.toml) | bounded Todd-Coxeter subgroup indices, finiteness tests, and canonical coset permutations for one-varying-relator presentations in the `words_and_growth` equation encoding | `generic` (deterministic Felsch tables with two-sided scans, portable C++) |
| [designs](modules/designs/manifest.toml) | whole-family tests and data for finite block designs: t-design multiplicities, resolutions, dual 2-designs, intersections, and Kramer-Mesner matrices | `generic` (portable C++) |
| [difference_sets](modules/difference_sets/manifest.toml) | difference sets, difference multisets, regular partial difference sets, and relative difference sets in finite permutation groups | `generic` (depth-first multiplicity pruning, portable C++) |
| [elliptic_curves_fp](modules/elliptic_curves_fp/manifest.toml) | curves `y^2 = x^3 + ax + b` over `F_p` (`p > 3`) from families of `(a, b)` pairs: point counts, singularity, supersingularity, j-invariants, `F_p`-isomorphism classes and the invariant factors of `E(F_p)` | `generic` (per-request square-root tables, per-member third differences, portable C++) |
| [graph_polynomials](modules/graph_polynomials/manifest.toml) | chromatic, Tutte, adjacency characteristic, and matching polynomials of labelled simple graphs drawn from edge-subset families | `generic` (memoized deletion-contraction and subset determinant DP, portable C++) |
| [gfp](modules/gfp/manifest.toml) | linear algebra over F_p on families of matrices: rank, rref, nullspace, span membership, solve, inverse, rref witness | `generic` (any p < 2^32, portable C++) |
| [gfq](modules/gfq/manifest.toml) | linear algebra over explicitly presented GF(q): rank, rref, nullspace, span membership, solve, inverse | `generic` (polynomial-basis arithmetic for q < 2^32, portable C++) |
| [graph_iso](modules/graph_iso/manifest.toml) | canonical forms, canonical vertex orders, and complete automorphism groups of undirected F_2 adjacency matrices, with loops allowed | `generic` (Weisfeiler-Lehman refinement and individualisation, portable C++) |
| [graphs](modules/graphs/manifest.toml) | connectivity, girth, diameter, colouring, clique and independence numbers, bipartiteness, degree sequences, and canonical forms of simple graphs | `generic` (portable C++; exact searches for colouring, cliques, and canonical labelling) |
| [hadamard](modules/hadamard/manifest.toml) | Hadamard, skew, regular and conference predicates for F_2 matrices read as signs, plus signed-equivalence canonical forms | `generic` (portable C++) |
| [heat_dirichlet](modules/heat_dirichlet/manifest.toml) | rigorous fixed-point upper bounds on the summands of heat-weighted Dirichlet polynomials (the de Bruijn-Newman barrier and canopy sums): the weight b_n n^(-sigma), the mollified summand enclosed over a cell of heights and cutoffs, that summand summed over blocks of consecutive n, and the Re s lower bound they consume; every rounding directed, the same integer in every implementation | `generic` (128-bit fixed point at scale 2^48, portable C++) |
| [hypergraphs](modules/hypergraphs/manifest.toml) | uniform hypergraphs: linearity, weak colouring number, Berge cycles and girth, finite Turan searches, and two-colour Ramsey searches | `generic` (bit sets and backtracking, portable C++) |
| [integer_partitions](modules/integer_partitions/manifest.toml) | constrained partitions and compositions: number of parts, largest part, and rank/crank distributions | `generic` (portable C++ enumeration) |
| [latin_squares](modules/latin_squares/manifest.toml) | Latin-square recognition, orthogonal mates, transversals, group tables, and isotopy canonical forms | `generic` (orders 1 through 5, portable C++) |
| [lattice_of_subspaces](modules/lattice_of_subspaces/manifest.toml) | Gaussian binomials, flag counts, counts above and below a row space, and subspace incidence predicates | `generic` (checked formulas and row reduction, portable C++) |
| [lattices_small](modules/lattices_small/manifest.toml) | positive-definite integral lattices via signed Gram matrices: minimum, kissing number, theta-series prefixes, parity, unimodularity, short vectors, and index-n sublattices | `generic` (Fincke-Pohst enumeration, exact norm and determinant checks, portable C++) |
| [linear_codes](modules/linear_codes/manifest.toml) | q-ary linear codes from generator matrices: minimum distance, weight enumerator, dual, self-duality, covering radius, MDS test, and coordinate-permutation automorphism order | `generic` (q-ary Gray-code word enumeration, portable C++) |
| [matrix_groups](modules/matrix_groups/manifest.toml) | order, irreducibility, absolute irreducibility, invariant bilinear forms, and imprimitivity for groups generated by subsets of a matrix dictionary | `generic` (Schreier stabilizer chain and finite-field searches, portable C++) |
| [matchings_and_flows](modules/matchings_and_flows/manifest.toml) | weighted bipartite perfect-matching counts, weighted undirected spanning-tree counts, and directed maximum-flow values from integer matrices | `generic` (Ryser permanent, matrix-tree determinant, and Edmonds-Karp, portable C++) |
| [orbits](modules/orbits/manifest.toml) | finite groups acting on families: orbit representatives (`is_canonical`), canonical index, orbit size, stabiliser order, fixed points for Burnside, and the projective action that turns a matrix group into permutations of points | `generic` (per-member orbit search, portable C++) |
| [perm_groups](modules/perm_groups/manifest.toml) | Schreier-Sims for permutation groups: order, membership, transitivity, primitivity, point-orbit partitions, and deterministic bases with strong generators | `generic` (portable C++, parallel across groups) |
| [permutation_statistics](modules/permutation_statistics/manifest.toml) | inversions, descents, major index, dense complete-cycle-type codes, classical pattern avoidance, and strong Bruhat order on permutation families | `generic` (parallel scans, pruned pattern matching, portable C++) |
| [polynomials_fq](modules/polynomials_fq/manifest.toml) | univariate polynomials over `F_q` on families of coefficient rows: irreducibility, factorisation degrees, primitivity, the order of `x`, roots and gcd | `generic` (Frobenius distinct-degree factorisation, portable C++) |
| [polytopes_small](modules/polytopes_small/manifest.toml) | vertex counts, f-vectors, simpliciality, and canonical h-star representations of Ehrhart polynomials for small lattice polytopes | `generic` (exact supporting-facet and lattice-point enumeration, portable C++) |
| [posets](modules/posets/manifest.toml) | finite posets from relation matrices, subset inclusion, or divisors: Möbius matrices, linear extensions, lattice and distributivity tests, width, height, and order-polynomial values | `generic` (exact dynamic programming, matching, and relation scans; portable C++) |
| [projective_sets](modules/projective_sets/manifest.toml) | arcs, caps, blocking sets, hyperovals, ovoids, span rank, maximum collinearity, and secant, tangent and passant counts for point sets in `PG(n,p)` | `generic` (precomputed line and hyperplane incidence, portable C++) |
| [quadratic_forms](modules/quadratic_forms/manifest.toml) | rank, canonical radicals, hyperbolic/elliptic/parabolic type, Witt index, exact isometry, and isotropic point counts for symmetric forms over odd prime fields | `generic` (symmetric congruence elimination and row reduction, portable C++) |
| [quantum_codes](modules/quantum_codes/manifest.toml) | binary symplectic stabiliser codes, equivalently additive GF(4) codes: self-orthogonality, distance, and the CSS property | `generic` (bit-packed elimination and increasing-weight error search, portable C++) |
| [residues](modules/residues/manifest.toml) | arithmetic in Z/n over ranges of residues: multiplicative order, primitive roots, quadratic residues, discrete logarithms by baby-step giant-step, the Legendre and Jacobi symbols, and the least primitive root of each modulus in a range | `generic` (factorisation of the modulus shared by every member, portable C++) |
| [set_systems](modules/set_systems/manifest.toml) | extremal predicates on finite set systems: intersecting families, antichains, sunflowers, maximum degree, lower shadows, and exact EKR/Sperner extremality | `generic` (prefix-pruned enumeration, portable C++) |
| [sieve_ranges](modules/sieve_ranges/manifest.toml) | arithmetic functions of every number in an interval: primality, prime factorisation, phi, sigma, tau, mu, omega, Omega and the largest prime factor, with Mathlib's conventions at 0 and 1 | `generic` (segmented sieve, portable C++) |
| [simplicial_complexes](modules/simplicial_complexes/manifest.toml) | abstract simplicial complexes read off 0/1 matrices, either generated by the rows or as the independence complex of the forbidden rows: f-vector, total faces, Euler characteristic, Betti numbers over F_p, and Bjorner-Wachs shellability | `generic` (faces as 64-bit masks, boundary elimination as bitsets over F_2 and dense over F_p, shelling search over facet subsets, portable C++) |
| [small_groups](modules/small_groups/manifest.toml) | invariants of a finite group given as a multiplication table: order, exponent, centre, conjugacy classes, derived series and length, nilpotence, solvability, and the number of subgroups and of normal subgroups | `generic` (subgroups as bit-set masks grown one coset at a time, portable C++) |
| [spreads_and_partitions](modules/spreads_and_partitions/manifest.toml) | sets of subspaces of F_p^n that miss each other: partial spreads, spreads, vector space partitions of mixed dimension, how many pairs of a set meet, and packings (parallelisms) whose rows are disjoint spreads | `generic` (incremental echelon per prefix, portable C++) |
| [strongly_regular](modules/strongly_regular/manifest.toml) | strongly regular graph parameters, exact spectra, and Krein and absolute-bound checks on adjacency-matrix families | `generic` (bit-packed common-neighbour tests, portable C++) |
| [subgroups](modules/subgroups/manifest.toml) | subgroup counts, conjugacy-class representatives, maximal subgroups, and normality for finite permutation groups | `generic` (cyclic extension over indexed multiplication tables, portable C++) |
| [subspace_orbits](modules/subspace_orbits/manifest.toml) | GL, PGL and PGammaL orbits of row spaces from Grassmannians and their transform/stack derivatives; canonical indices use the least rref in Grassmannian order | `generic` (per-member rref orbit search, portable C++) |
| [sum_free_and_additive](modules/sum_free_and_additive/manifest.toml) | sum-free, Sidon and k-AP-free sets of integers or of Z/n, with sumset and difference-set sizes, Schur triple counts, difference multiplicities and a doubling-constant test | `generic` (forbidden-set bitsets, span table, mirror rule and gap bounds for Golomb-ruler and r_k(n) searches, portable C++) |
| [vertex_transitive](modules/vertex_transitive/manifest.toml) | vertex and arc transitivity, Cayley recognition, and canonical lists of regular automorphism subgroups of simple graphs | `generic` (colour-refined automorphism search and direct semiregular subgroup generation, portable C++) |
| [words_and_growth](modules/words_and_growth/manifest.toml) | ball sizes, spherical growth-series coefficients, and geodesic-word counts for finite complete rewriting presentations of groups | `generic` (critical-pair validation and normal-form automata, portable C++) |
| [young](modules/young/manifest.toml) | partitions and standard Young tableaux, hook-length counts, Kostka numbers, RSK, and irreducible S_n character values by Murnaghan-Nakayama | `generic` (portable C++) |

`ctx.group_tables(lk.naturals(tables))` builds a family from stored Cayley tables;
`ctx.generated_group(permutations)` builds one from permutation generators;
`ctx.group_catalogue(n)` builds one from the stored catalogue of every group of order `n` for
`n` up to 31, one member per isomorphism class, in increasing lexicographic order of the tables
that `tools/group_catalogue.py` canonicalised.
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
(on the right, `M ↦ M A`). [BENCHMARKS.md](BENCHMARKS.md) has every module's generic backend
against the naive Python implementation on inputs small enough for naive to finish, so every row
is a byte-for-byte agreement. Single-threaded ratios run from tens (Grassmannian orbits, where
each step is an elimination) to thousands (independence of subsets; isomorphism classes of
elliptic curves), and beyond that where the backend changes the algorithm (Burnside counts from
cycle types; sieving an interval once instead of factoring each of its members on its own) or
refuses to look at most of the family at all (spreads: one meeting pair kills every extension of
that prefix; sum-free subsets are the same trick).

`elliptic_curves_fp` reads a `1 x 2` member as the pair `(a, b)` of `y^2 = x^3 + ax + b`, so
`all_matrices(p, 1, 2)` is every curve over `F_p` and a histogram of `point_count` over it is the
isogeny classes; families of another shape are refused rather than reinterpreted.

A member of a spread question is read as a stack of components: with `n` given, each row of the
member is `cols / n` vectors of F_p^n and spans one subspace, so a family of `k`-subsets of the
Grassmannian is a family of `k` candidate components. A packing reads each row as a whole spread
instead, splitting it into `cols / (n * h)` components of `h` vectors.

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

`code_equivalence` reads a member as a generator matrix and asks which codes it cannot be told
apart from. The group is the monomial group of `F_q^n`, of order `(q-1)^n n!`: `scalars=1` allows
a nonzero scalar on each coordinate as well as permuting them, `scalars=0` permutes only, and
over `F_2` the two coincide. A class is named by its least code in Grassmannian order, which is
always systematic, so `canonical_form` returns `[I_k | A]` with `A` the least row-major digit
string in the class. Counting `is_canonical` over `grassmannian(p, n, k)` is the number of
inequivalent `[n,k]` codes: 22 binary `[6,3]` codes, 43 `[7,3]`, and 705 `[10,5]` from the 109
million subspaces of `F_2^10`, in 14 seconds.
`sieve_ranges` reads a `1 x 1` natural-number member as the number itself, so `range(a, b)` is an
interval and `explicit` is any list of numbers. It follows Mathlib at the ends: 0 and 1 have the
empty factorisation, phi(0) = sigma(0) = tau(0) = 0 and phi(1) = sigma(1) = tau(1) = 1. Möbius mu
is returned shifted as mu(n) + 1 in {0, 1, 2}, because a reduction carries unsigned values, so a
histogram is (#mu=-1, #mu=0, #mu=+1) and `sum` minus the family size is the Mertens function of
the interval.

`subspace_orbits` treats each matrix as a row space, so it also handles transforms and stacks that
change the displayed basis or its rank. Set `projective=0` to retain the generated GL group order,
or `projective=1` to identify nonzero scalar matrices. The runtime currently works over prime
fields, where PGammaL has the same action as PGL.

To add to this table, read [docs/adding-a-module.md](docs/adding-a-module.md). To make an
existing module faster, read [docs/adding-a-backend.md](docs/adding-a-backend.md).

## Where it lives

Source: `~/projects/LemmaKernel`, public remote `hara-seihun/LemmaKernel`. It is not yet
deployed anywhere outside the checkout; the shared-library and `/srv/pi` publication step is open.
