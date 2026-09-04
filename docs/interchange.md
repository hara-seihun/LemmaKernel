# Interchange encoding

Every object has exactly one encoding, used by `lk_import`/`lk_export`, by the Python binding,
by the naive implementations, by tests, and by anything written to disk. It is deliberately dumb
so that a reader in any language can parse it in an afternoon.

```
magic     "LKIF"
u32       format version (2)
string    kind                      u32 byte length, then UTF-8
u32       parameter count
repeated  string name, u128 value   parameters, sorted by name
u64       payload length
bytes     payload
```

All integers little-endian; a u128 is its low u64 then its high u64. Parameters are 128-bit
because family sizes, member indices and counts over a family exceed 2^64 once a search prunes
(the runtime settles families of 10^20 members); every other parameter is a 64-bit quantity
stored in the low half. Kinds, their parameters, and their payload layouts are declared in
each module's manifest (`[[kinds]]`) and in the runtime's own kinds below. Field entries are labels
in `0..q-1`; the existing `p` header parameter carries this field-size tag. The `gfp` module requires
it to be prime, while `gfq` supplies an explicit extension-field presentation. Entries use the
smallest width holding q−1: 1 byte for q < 2^8, 2 for q < 2^16, 4 for q < 2^32, 8 otherwise.
Permutations (`orbits.perms`, p = 0) and naturals (`lk.naturals`, p = 2^64 - 1 in a result header)
are packed as 4-byte values. Signed integral matrices (`lattices.gram`, p = 2^64 - 2 in a result
header) use 4-byte ZigZag values: z >= 0 maps to 2z, and z < 0 maps to -2z-1.

Runtime kinds (not module-specific):

| kind | params | payload |
|---|---|---|
| `integers` | count | u64[count] |
| `count` | value, visited, family_size | empty |
| `histogram` | visited, family_size, bins | u64[bins] |
| `hits` | p, rows, cols, total, visited, family_size, count, materialised | u128 indices[count], then materialised·rows·cols entries |
| `first` | p, rows, cols, found, index, visited, family_size | found·rows·cols entries (the hit, if any) |
| `extremum` | p, rows, cols, value, index, visited, family_size | rows·cols entries (the member attaining the value) |
| `lk.naturals` | count, rows, cols | count·rows·cols 4-byte entries |
| `lk.signed_matrices` | count, rows, cols | count·rows·cols signed 8-byte entries |
| `family.<name>` | the family's integer parameters | the nested objects (matrices, inner family) as concatenated blobs, in the order the manifest lists them |

`sum` returns kind `count` with the sum as its value; per-member values are u64 but the sum is a full u128 parameter. `first` reports `visited` as index + 1 when
found and family_size otherwise, so the encoding does not depend on how many members other
threads happened to decide.

A family blob is small (a Grassmannian is three integers) and re-importable, so a family can be
saved, sent, and reproduced exactly.

`python/lemmakernel/interchange.py` is the Python reader and writer; `runtime/src/object.cpp` the
C++ one. They are tested against each other by round-tripping in the module tests.
