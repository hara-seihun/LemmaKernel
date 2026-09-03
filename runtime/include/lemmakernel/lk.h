/* LemmaKernel stable C ABI.
 *
 * This header is the whole native surface. Modules do not add functions to it; they register
 * operations by name, and callers discover them through lk_describe(). Objects are opaque handles
 * bound to the context that created them; data crosses the boundary only as interchange blobs
 * (see PHILOSOPHY.md and docs/interchange.md).
 */
#ifndef LEMMAKERNEL_LK_H
#define LEMMAKERNEL_LK_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct lk_context lk_context;
typedef uint64_t lk_handle; /* 0 is the null handle */

typedef enum lk_status {
    LK_OK = 0,
    LK_INVALID_ARGUMENT = 1, /* malformed input, unknown name, shape mismatch, bad handle */
    LK_UNSUPPORTED = 2,      /* no backend accepts this request */
    LK_OUT_OF_MEMORY = 3,
    LK_INTERNAL = 4          /* a backend violated its own invariants; report it */
} lk_status;

/* Context: backend selection, thread budget, last error. `backend` is NULL or "auto" for
 * automatic selection, or a backend name from lk_describe() (e.g. "gfp.generic") to force it for
 * that module's requests; other modules keep automatic selection. */
lk_status lk_context_create(const char *backend, lk_context **out);
void lk_context_destroy(lk_context *ctx);
lk_status lk_context_set_threads(lk_context *ctx, uint32_t threads);
/* Text of the last error on this context; valid until the next call on the context. */
const char *lk_context_error(const lk_context *ctx);

/* JSON description of every module compiled into this library: object kinds, families,
 * operations, reductions, backends and whether each backend is usable on this machine. Static. */
const char *lk_describe(void);

/* Objects. */
lk_status lk_import(lk_context *ctx, const uint8_t *bytes, size_t len, lk_handle *out);
lk_status lk_export(lk_context *ctx, lk_handle h, uint8_t **bytes, size_t *len); /* free with lk_free */
void lk_free(void *p);
lk_status lk_release(lk_context *ctx, lk_handle h);
lk_status lk_handle_kind(lk_context *ctx, lk_handle h, const char **kind);
lk_status lk_handle_param(lk_context *ctx, lk_handle h, const char *name, uint64_t *value);

/* Families (see manifest "families"). Constructed here rather than imported because their
 * descriptions are tiny; they can still be exported and re-imported as interchange blobs. */
lk_status lk_family_explicit(lk_context *ctx, lk_handle batch, lk_handle *out);
lk_status lk_family_subsets(lk_context *ctx, lk_handle dictionary, uint64_t k, lk_handle *out);
lk_status lk_family_grassmannian(lk_context *ctx, uint64_t p, uint64_t n, uint64_t h, lk_handle *out);
lk_status lk_family_all_matrices(lk_context *ctx, uint64_t p, uint64_t rows, uint64_t cols, lk_handle *out);
lk_status lk_family_transform(lk_context *ctx, lk_handle family, lk_handle matrix, lk_handle *out);
lk_status lk_family_stack(lk_context *ctx, lk_handle family, lk_handle rows, lk_handle *out);
lk_status lk_family_group_elements(lk_context *ctx, lk_handle generators, lk_handle *out);
/* Stored Cayley tables, or one group converted from permutation generators. */
lk_status lk_family_group_tables(lk_context *ctx, lk_handle tables, lk_handle *out);
lk_status lk_family_generated_group(lk_context *ctx, lk_handle generators, lk_handle *out);
lk_status lk_family_subsets_of(lk_context *ctx, lk_handle family, uint64_t k, lk_handle *out);
lk_status lk_family_symmetric_matrices(lk_context *ctx, uint64_t p, uint64_t n, lk_handle *out);
lk_status lk_family_range(lk_context *ctx, uint64_t a, uint64_t b, lk_handle *out);
lk_status lk_family_words(lk_context *ctx, uint64_t alphabet, uint64_t length, lk_handle *out);
lk_status lk_family_partitions(lk_context *ctx, uint64_t total, uint64_t max_part, uint64_t max_parts,
                               uint64_t max_multiplicity, uint64_t distinct, uint64_t odd, lk_handle *out);
lk_status lk_family_compositions(lk_context *ctx, uint64_t total, uint64_t parts, uint64_t max_part, lk_handle *out);
lk_status lk_family_standard_tableaux(lk_context *ctx, lk_handle shape, lk_handle *out);
lk_status lk_family_size(lk_context *ctx, lk_handle family, uint64_t *size);
lk_status lk_family_member(lk_context *ctx, lk_handle family, uint64_t index, lk_handle *out);

/* Operations. `op` and `reduction` are names from the manifest ("gfp.rank", "histogram").
 * Arguments are named; each is a handle or an integer as the manifest says. */
typedef struct lk_arg {
    const char *name;
    lk_handle handle;
    uint64_t integer;
} lk_arg;

lk_status lk_run(lk_context *ctx, const char *op, lk_handle family, const char *reduction,
                 const lk_arg *args, size_t nargs, lk_handle *out);

#ifdef __cplusplus
}
#endif
#endif
