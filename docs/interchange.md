# Interchange encoding

Every object has exactly one encoding, used by `lk_import`/`lk_export`, by the Python binding,
by the naive implementations, by tests, and by anything written to disk. It is deliberately dumb
so that a reader in any language can parse it in an afternoon.

```
magic     "LKIF"
u32       format version (1)
string    kind                      u32 byte length, then UTF-8
u32       parameter count
repeated  string name, u64 value    parameters, sorted by name
u64       payload length
bytes     payload
```

All integers little-endian. Kinds, their parameters, and their payload layouts are declared in
each module's manifest (`[[kinds]]`) and in the runtime's own kinds below. Entries over F_p are
packed at the smallest width holding p−1: 1 byte for p < 2^8, 2 for p < 2^16, 4 for p < 2^32,
8 otherwise.

Runtime kinds (not module-specific):

| kind | params | payload |
|---|---|---|
| `integers` | count | u64[count] |
| `count` | value, visited, family_size | empty |
| `histogram` | visited, family_size, bins | u64[bins] |
| `hits` | p, rows, cols, total, visited, family_size, count, materialised | u64 indices[count], then materialised·rows·cols entries |
| `family.<name>` | the family's integer parameters | the nested objects (matrices, inner family) as concatenated blobs, in the order the manifest lists them |

A family blob is small (a Grassmannian is three integers) and re-importable, so a family can be
saved, sent, and reproduced exactly.

`python/lemmakernel/interchange.py` is the Python reader and writer; `runtime/src/object.cpp` the
C++ one. They are tested against each other by round-tripping in the module tests.
