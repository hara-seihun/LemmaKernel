#include "lemmakernel/lk.h"

#include "family.hpp"
#include "object.hpp"
#include "registry.hpp"

#include <cstring>
#include <mutex>
#include <thread>
#include <unordered_map>

#if defined(__x86_64__)
#include <cpuid.h>
#endif

namespace lk {

std::vector<Backend> &backend_store() {
    static std::vector<Backend> v;
    return v;
}
void register_backend(Backend b) { backend_store().push_back(std::move(b)); }
const std::vector<Backend> &backends() { return backend_store(); }

bool cpu_has(const char *feature) {
#if defined(__x86_64__)
    unsigned a, b, c, d;
    if (!__get_cpuid_count(7, 0, &a, &b, &c, &d)) return false;
    if (std::strcmp(feature, "avx512bw") == 0) return (b & (1u << 30)) && (b & (1u << 16));
#endif
    return false;
}

} // namespace lk

using namespace lk;

struct lk_context {
    std::string backend_selector;
    uint32_t threads = std::thread::hardware_concurrency() ? std::thread::hardware_concurrency() : 1;
    std::string error;
    std::unordered_map<lk_handle, std::shared_ptr<Object>> objects;
    lk_handle next = 1;

    lk_status set_error(int status, std::string msg) {
        error = std::move(msg);
        return (lk_status)status;
    }
    lk_status set_error(const Error &e) { return set_error(e.status, e.message); }
    lk_handle put(std::shared_ptr<Object> o) {
        lk_handle h = next++;
        objects[h] = std::move(o);
        return h;
    }
    Result<std::shared_ptr<Object>> get(lk_handle h) {
        auto it = objects.find(h);
        if (it == objects.end()) return Result<std::shared_ptr<Object>>::failure(LK_INVALID_ARGUMENT, "unknown handle");
        return Result<std::shared_ptr<Object>>::success(it->second);
    }
    Result<std::shared_ptr<Matrix>> get_matrix(lk_handle h, const char *what) {
        auto o = get(h);
        if (!o.ok) return Result<std::shared_ptr<Matrix>>::failure(o.error.status, o.error.message);
        if (!o.value->matrix) return Result<std::shared_ptr<Matrix>>::failure(LK_INVALID_ARGUMENT, std::string(what) + " must be a gfp.matrix");
        return Result<std::shared_ptr<Matrix>>::success(o.value->matrix);
    }
    Result<std::shared_ptr<Family>> get_family(lk_handle h) {
        auto o = get(h);
        if (!o.ok) return Result<std::shared_ptr<Family>>::failure(o.error.status, o.error.message);
        if (!o.value->family) return Result<std::shared_ptr<Family>>::failure(LK_INVALID_ARGUMENT, "handle is not a family");
        return Result<std::shared_ptr<Family>>::success(o.value->family);
    }
};

static std::shared_ptr<Object> family_object(std::shared_ptr<Family> f) {
    auto o = std::make_shared<Object>();
    o->kind = std::string("family.") + family_kind_name(f->kind);
    o->family = std::move(f);
    return o;
}

extern "C" {

lk_status lk_context_create(const char *backend, lk_context **out) {
    if (!out) return LK_INVALID_ARGUMENT;
    auto *ctx = new lk_context();
    ctx->backend_selector = (backend && std::strcmp(backend, "auto") != 0) ? backend : "";
    if (!ctx->backend_selector.empty()) {
        bool known = false;
        for (const auto &b : backends()) known |= (b.module + "." + b.name) == ctx->backend_selector;
        if (!known) {
            delete ctx;
            return LK_INVALID_ARGUMENT;
        }
    }
    *out = ctx;
    return LK_OK;
}

void lk_context_destroy(lk_context *ctx) { delete ctx; }

lk_status lk_context_set_threads(lk_context *ctx, uint32_t threads) {
    if (!ctx) return LK_INVALID_ARGUMENT;
    if (threads == 0) return ctx->set_error(LK_INVALID_ARGUMENT, "threads must be >= 1");
    ctx->threads = threads;
    return LK_OK;
}

const char *lk_context_error(const lk_context *ctx) { return ctx ? ctx->error.c_str() : ""; }

const char *lk_describe(void) {
    static std::string json;
    static std::once_flag once;
    std::call_once(once, [] {
        std::string base = manifest_describe_json();
        std::string avail = ",\"available_backends\":[";
        bool first = true;
        for (const auto &b : backends()) {
            if (!b.available()) continue;
            avail += (first ? "\"" : ",\"") + b.module + "." + b.name + "\"";
            first = false;
        }
        avail += "]";
        json = base.substr(0, base.size() - 1) + avail + "}";
    });
    return json.c_str();
}

lk_status lk_import(lk_context *ctx, const uint8_t *bytes, size_t len, lk_handle *out) {
    if (!ctx || !bytes || !out) return LK_INVALID_ARGUMENT;
    auto r = decode(bytes, len);
    if (!r.ok) return ctx->set_error(r.error);
    *out = ctx->put(r.value);
    return LK_OK;
}

lk_status lk_export(lk_context *ctx, lk_handle h, uint8_t **bytes, size_t *len) {
    if (!ctx || !bytes || !len) return LK_INVALID_ARGUMENT;
    auto o = ctx->get(h);
    if (!o.ok) return ctx->set_error(o.error);
    std::vector<uint8_t> enc = encode(*o.value);
    auto *buf = (uint8_t *)std::malloc(enc.size() ? enc.size() : 1);
    if (!buf) return ctx->set_error(LK_OUT_OF_MEMORY, "malloc failed");
    std::memcpy(buf, enc.data(), enc.size());
    *bytes = buf;
    *len = enc.size();
    return LK_OK;
}

void lk_free(void *p) { std::free(p); }

lk_status lk_release(lk_context *ctx, lk_handle h) {
    if (!ctx) return LK_INVALID_ARGUMENT;
    if (!ctx->objects.erase(h)) return ctx->set_error(LK_INVALID_ARGUMENT, "unknown handle");
    return LK_OK;
}

lk_status lk_handle_kind(lk_context *ctx, lk_handle h, const char **kind) {
    if (!ctx || !kind) return LK_INVALID_ARGUMENT;
    auto o = ctx->get(h);
    if (!o.ok) return ctx->set_error(o.error);
    *kind = o.value->kind.c_str();
    return LK_OK;
}

lk_status lk_handle_param(lk_context *ctx, lk_handle h, const char *name, uint64_t *value) {
    if (!ctx || !name || !value) return LK_INVALID_ARGUMENT;
    auto o = ctx->get(h);
    if (!o.ok) return ctx->set_error(o.error);
    auto params = o.value->params();
    auto it = params.find(name);
    if (it == params.end()) return ctx->set_error(LK_INVALID_ARGUMENT, std::string("no parameter ") + name + " on " + o.value->kind);
    *value = it->second;
    return LK_OK;
}

#define FAMILY_RESULT(expr)                                                                        \
    do {                                                                                           \
        auto fr = (expr);                                                                          \
        if (!fr.ok) return ctx->set_error(fr.error);                                               \
        *out = ctx->put(family_object(fr.value));                                                  \
        return LK_OK;                                                                              \
    } while (0)

lk_status lk_family_explicit(lk_context *ctx, lk_handle batch, lk_handle *out) {
    if (!ctx || !out) return LK_INVALID_ARGUMENT;
    auto m = ctx->get_matrix(batch, "batch");
    if (!m.ok) return ctx->set_error(m.error);
    FAMILY_RESULT(make_explicit(m.value));
}

lk_status lk_family_subsets(lk_context *ctx, lk_handle dictionary, uint64_t k, lk_handle *out) {
    if (!ctx || !out) return LK_INVALID_ARGUMENT;
    auto m = ctx->get_matrix(dictionary, "dictionary");
    if (!m.ok) return ctx->set_error(m.error);
    FAMILY_RESULT(make_subsets(m.value, k));
}

lk_status lk_family_grassmannian(lk_context *ctx, uint64_t p, uint64_t n, uint64_t h, lk_handle *out) {
    if (!ctx || !out) return LK_INVALID_ARGUMENT;
    FAMILY_RESULT(make_grassmannian(p, n, h));
}

lk_status lk_family_all_matrices(lk_context *ctx, uint64_t p, uint64_t rows, uint64_t cols, lk_handle *out) {
    if (!ctx || !out) return LK_INVALID_ARGUMENT;
    FAMILY_RESULT(make_all_matrices(p, rows, cols));
}

lk_status lk_family_transform(lk_context *ctx, lk_handle family, lk_handle matrix, lk_handle *out) {
    if (!ctx || !out) return LK_INVALID_ARGUMENT;
    auto f = ctx->get_family(family);
    if (!f.ok) return ctx->set_error(f.error);
    auto m = ctx->get_matrix(matrix, "transform matrix");
    if (!m.ok) return ctx->set_error(m.error);
    FAMILY_RESULT(make_transform(f.value, m.value));
}

lk_status lk_family_stack(lk_context *ctx, lk_handle family, lk_handle rows, lk_handle *out) {
    if (!ctx || !out) return LK_INVALID_ARGUMENT;
    auto f = ctx->get_family(family);
    if (!f.ok) return ctx->set_error(f.error);
    auto m = ctx->get_matrix(rows, "stacked rows");
    if (!m.ok) return ctx->set_error(m.error);
    FAMILY_RESULT(make_stack(f.value, m.value));
}

lk_status lk_family_size(lk_context *ctx, lk_handle family, uint64_t *size) {
    if (!ctx || !size) return LK_INVALID_ARGUMENT;
    auto f = ctx->get_family(family);
    if (!f.ok) return ctx->set_error(f.error);
    auto s = f.value->size();
    if (!s.ok) return ctx->set_error(s.error);
    *size = s.value;
    return LK_OK;
}

lk_status lk_family_member(lk_context *ctx, lk_handle family, uint64_t index, lk_handle *out) {
    if (!ctx || !out) return LK_INVALID_ARGUMENT;
    auto f = ctx->get_family(family);
    if (!f.ok) return ctx->set_error(f.error);
    auto m = f.value->member(index);
    if (!m.ok) return ctx->set_error(m.error);
    auto o = std::make_shared<Object>();
    o->kind = matrix_kind(m.value);
    o->matrix = std::make_shared<Matrix>(std::move(m.value));
    *out = ctx->put(o);
    return LK_OK;
}

lk_status lk_family_group_elements(lk_context *ctx, lk_handle generators, lk_handle *out) {
    if (!ctx || !out) return LK_INVALID_ARGUMENT;
    auto o = ctx->get(generators);
    if (!o.ok) return ctx->set_error(o.error);
    if (!o.value->matrix || o.value->matrix->p != 0) return ctx->set_error(LK_INVALID_ARGUMENT, "generators must be an orbits.perms batch");
    FAMILY_RESULT(make_group_elements(o.value->matrix));
}

lk_status lk_run(lk_context *ctx, const char *op, lk_handle family, const char *reduction,
                 const lk_arg *args, size_t nargs, lk_handle *out) {
    if (!ctx || !op || !reduction || !out) return LK_INVALID_ARGUMENT;
    std::string full = op;
    auto dot = full.find('.');
    if (dot == std::string::npos) return ctx->set_error(LK_INVALID_ARGUMENT, "operation must be module.name");
    Request req;
    req.module = full.substr(0, dot);
    req.op = full.substr(dot + 1);
    req.reduction = reduction;
    req.threads = ctx->threads;

    const ManifestOperation *mop = nullptr;
    for (const auto &o : manifest_operations())
        if (req.module == o.module && req.op == o.name) mop = &o;
    if (!mop) return ctx->set_error(LK_INVALID_ARGUMENT, "unknown operation " + full);
    req.value_type = mop->value;
    const ManifestReduction *mred = nullptr;
    for (const auto &r : manifest_reductions())
        if (req.reduction == r.name) mred = &r;
    if (!mred) return ctx->set_error(LK_INVALID_ARGUMENT, "unknown reduction " + req.reduction);
    bool accepted = false;
    for (const char *const *a = mred->accepts; *a; ++a) accepted |= req.value_type == *a || std::string("*") == *a;
    if (!accepted) return ctx->set_error(LK_INVALID_ARGUMENT, "reduction " + req.reduction + " does not accept " + req.value_type + " values (operation " + full + ")");

    auto f = ctx->get_family(family);
    if (!f.ok) return ctx->set_error(f.error);
    req.family = f.value;
    if (*mop->families) {
        bool allowed = false;
        std::string allowed_names;
        for (const char *const *fk = mop->families; *fk; ++fk) {
            allowed |= std::string(family_kind_name(f.value->kind)) == *fk;
            allowed_names += (allowed_names.empty() ? "" : ", ") + std::string(*fk);
        }
        if (!allowed) return ctx->set_error(LK_INVALID_ARGUMENT, full + " is defined on " + allowed_names + " families only");
    }

    std::vector<std::string> expected;
    for (const char *const *a = mop->args; *a; ++a) expected.push_back(*a);
    for (const char *const *a = mred->args; *a; ++a) expected.push_back(*a);
    for (size_t i = 0; i < nargs; ++i) {
        std::string name = args[i].name ? args[i].name : "";
        bool known = false;
        for (const auto &e : expected) known |= e == name;
        if (!known) return ctx->set_error(LK_INVALID_ARGUMENT, "unexpected argument " + name + " for " + full + "/" + req.reduction);
        if (args[i].handle) {
            auto o = ctx->get(args[i].handle);
            if (!o.ok) return ctx->set_error(o.error);
            req.handle_args[name] = o.value;
        } else {
            req.int_args[name] = args[i].integer;
        }
    }
    for (const auto &e : expected)
        if (!req.handle_args.count(e) && !req.int_args.count(e))
            return ctx->set_error(LK_INVALID_ARGUMENT, "missing argument " + e + " for " + full + "/" + req.reduction);

    /* A pinned backend applies to requests for its own module; other modules select as usual. */
    bool pinned = !ctx->backend_selector.empty() && ctx->backend_selector.rfind(req.module + ".", 0) == 0;
    const Backend *chosen = nullptr;
    for (const auto &b : backends()) {
        if (b.module != req.module) continue;
        std::string id = b.module + "." + b.name;
        if (pinned && id != ctx->backend_selector) continue;
        if (!b.available() || !b.accepts(req)) continue;
        if (!chosen || b.specificity > chosen->specificity) chosen = &b;
    }
    if (!chosen) {
        if (pinned)
            return ctx->set_error(LK_UNSUPPORTED, "backend " + ctx->backend_selector + " does not accept this request");
        return ctx->set_error(LK_UNSUPPORTED, "no available backend accepts " + full + " on this family");
    }
    auto r = chosen->run(req);
    if (!r.ok) return ctx->set_error(r.error);
    *out = ctx->put(r.value);
    return LK_OK;
}

} // extern "C"
