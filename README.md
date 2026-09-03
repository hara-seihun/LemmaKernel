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
  group, and `range`/`words` of natural numbers). Family sizes in the hundreds of millions are
  normal.
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
| [burnside](modules/burnside/manifest.toml) | orbit counts, fixed counts, and canonical cycle indices for permutation groups on subsets and words, without enumerating the acted-on family | `generic` (cycle-type formulas, portable C++) |
| [cayley](modules/cayley/manifest.toml) | simple undirected Cayley graphs of finite permutation groups: connectivity, regularity, girth, diameter, graph automorphism order, and the CI property of a connection set | `generic` (exact group closure, graph search, partition refinement and individualisation, portable C++) |
| [gfp](modules/gfp/manifest.toml) | linear algebra over F_p on families of matrices: rank, rref, nullspace, span membership, solve, inverse, rref witness | `generic` (any p < 2^32, portable C++) |
| [orbits](modules/orbits/manifest.toml) | finite groups acting on families: orbit representatives (`is_canonical`), canonical index, orbit size, stabiliser order, fixed points for Burnside, and the projective action that turns a matrix group into permutations of points | `generic` (per-member orbit search, portable C++) |
| [projective_sets](modules/projective_sets/manifest.toml) | arcs, caps, blocking sets, hyperovals, ovoids, span rank, maximum collinearity, and secant, tangent and passant counts for point sets in `PG(n,p)` | `generic` (precomputed line and hyperplane incidence, portable C++) |
| [subspace_orbits](modules/subspace_orbits/manifest.toml) | GL, PGL and PGammaL orbits of row spaces from Grassmannians and their transform/stack derivatives; canonical indices use the least rref in Grassmannian order | `generic` (per-member rref orbit search, portable C++) |

An orbit is a permutation of member indices; the representative is the least index. Permutation
groups act on `subsets` families, matrix groups on `grassmannian` and `all_matrices` families
(on the right, `M ↦ M A`). `tools/bench.py` compares each module's generic backend with the
naive Python implementation on inputs small enough for naive to finish, so every row is a
byte-for-byte agreement; single-threaded ratios on this machine run from about 40× (Grassmannian
orbits, where each step is an elimination) to about 1700× (independence of subsets), and higher
where the backend changes the algorithm (Burnside counts from cycle types).

`subspace_orbits` treats each matrix as a row space, so it also handles transforms and stacks that
change the displayed basis or its rank. Set `projective=0` to retain the generated GL group order,
or `projective=1` to identify nonzero scalar matrices. The runtime currently works over prime
fields, where PGammaL has the same action as PGL.

To add to this table, read [docs/adding-a-module.md](docs/adding-a-module.md). To make an
existing module faster, read [docs/adding-a-backend.md](docs/adding-a-backend.md).

## Where it lives

Source: `~/projects/LemmaKernel`, public remote `hara-seihun/LemmaKernel`. It is not yet
deployed anywhere outside the checkout; the shared-library and `/srv/pi` publication step is open.
