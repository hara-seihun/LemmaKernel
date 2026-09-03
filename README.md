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
```

`lk.describe()` lists every module compiled in: its object kinds, families, operations,
reductions, backends, and which backends run on this machine. The Python binding is a thin
`ctypes` layer over the C ABI in [`runtime/include/lemmakernel/lk.h`](runtime/include/lemmakernel/lk.h);
anything Python can do, C can do with the same names.

Vocabulary, in the order a request is built:

- **family**: a description of a set of matrices with a canonical order (`explicit`, `subsets`,
  `grassmannian`, `all_matrices`, and `transform`/`stack` wrappers). Family sizes in the
  hundreds of millions are normal.
- **operation**: what to compute per member (`gfp.rank`, `gfp.in_span`, `gfp.rref`, ...).
- **reduction**: what to bring back (`count`, `histogram`, `hits`, or `all`).

A `count`, `histogram`, or `hits` result carries `visited` and `family_size`; they are equal or
the call fails. That is the completeness check for a non-existence answer.

## Build and test

```
cmake -S . -B build -G Ninja && ninja -C build      # liblemmakernel.so
pytest -n auto modules                              # every module's tests
modules/gfp/bench/bench.py                          # kernel vs naive, with headline speed-ups
```

The Lean side (`lake build`) needs Mathlib; `.lake/packages` is a hardlink copy of
`~/projects/LemmaLib/.lake/packages`, which pins the same Lean and Mathlib versions. Tests only
need the module's `Reference` (no Mathlib import) and take a few seconds each.

## Layout

```
runtime/        the C ABI, object encoding, family enumeration, backend registry
modules/gfp/    one module: manifest, Lean contract and reference, backends, naive, tests, bench
python/         the binding; python/lemmakernel/_manifest.py is generated
tools/          manifest.py (generator and checker), leancheck.py (the test oracle)
docs/           how to add a module or a backend
```

Every derived file is produced by `tools/manifest.py generate` from `modules/*/manifest.toml`
and committed; the build refuses to configure if they are stale.

## Modules

| module | what it computes | backends |
|---|---|---|
| [gfp](modules/gfp/manifest.toml) | linear algebra over F_p on families of matrices: rank, rref, nullspace, span membership, solve, inverse, rref witness | `generic` (any p < 2^32, portable C++) |

To add to this table, read [docs/adding-a-module.md](docs/adding-a-module.md). To make an
existing module faster, read [docs/adding-a-backend.md](docs/adding-a-backend.md).

## Where it lives

Source: `~/projects/LemmaKernel`, public remote `hara-seihun/LemmaKernel`. It is not yet
deployed anywhere outside the checkout; the shared-library and `/srv/pi` publication step is open.
