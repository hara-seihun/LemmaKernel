#include "object.hpp"
#include "family.hpp"

#include <cstring>

namespace lk {

namespace {

constexpr int INVALID = 1;
constexpr uint32_t FORMAT_VERSION = 1;
const char MAGIC[4] = {'L', 'K', 'I', 'F'};

struct Writer {
    std::vector<uint8_t> out;
    void u32(uint32_t v) { for (int i = 0; i < 4; ++i) out.push_back((uint8_t)(v >> (8 * i))); }
    void u64(uint64_t v) { for (int i = 0; i < 8; ++i) out.push_back((uint8_t)(v >> (8 * i))); }
    void str(const std::string &s) { u32((uint32_t)s.size()); out.insert(out.end(), s.begin(), s.end()); }
    void entries(const std::vector<Entry> &e, unsigned width) {
        size_t start = out.size();
        out.resize(start + e.size() * width);
        uint8_t *dst = out.data() + start;
        for (Entry v : e) {
            for (unsigned i = 0; i < width; ++i) *dst++ = (uint8_t)((uint64_t)v >> (8 * i));
        }
    }
    void u64s(const std::vector<uint64_t> &v) { for (uint64_t x : v) u64(x); }
    void bytes(const std::vector<uint8_t> &b) { out.insert(out.end(), b.begin(), b.end()); }
};

struct Reader {
    const uint8_t *p, *end;
    bool bad = false;
    uint32_t u32() {
        if (end - p < 4) { bad = true; return 0; }
        uint32_t v = 0;
        for (int i = 0; i < 4; ++i) v |= (uint32_t)p[i] << (8 * i);
        p += 4;
        return v;
    }
    uint64_t u64() {
        if (end - p < 8) { bad = true; return 0; }
        uint64_t v = 0;
        for (int i = 0; i < 8; ++i) v |= (uint64_t)p[i] << (8 * i);
        p += 8;
        return v;
    }
    std::string str() {
        uint32_t n = u32();
        if (bad || (size_t)(end - p) < n) { bad = true; return {}; }
        std::string s((const char *)p, n);
        p += n;
        return s;
    }
    bool entries(std::vector<Entry> &out, uint64_t n, unsigned width) {
        if ((unsigned __int128)n * width > (uint64_t)(end - p)) { bad = true; return false; }
        out.resize(n);
        for (uint64_t i = 0; i < n; ++i) {
            uint64_t v = 0;
            for (unsigned b = 0; b < width; ++b) v |= (uint64_t)p[b] << (8 * b);
            p += width;
            out[i] = (Entry)v;
        }
        return true;
    }
};

struct Header {
    std::string kind;
    std::map<std::string, uint64_t> params;
    const uint8_t *payload;
    uint64_t payload_len;
    const uint8_t *next;
};

Result<Header> read_header(const uint8_t *bytes, size_t len) {
    if (len < 4 || std::memcmp(bytes, MAGIC, 4) != 0)
        return Result<Header>::failure(INVALID, "not an interchange blob (bad magic)");
    Reader r{bytes + 4, bytes + len};
    Header h;
    uint32_t version = r.u32();
    if (r.bad || version != FORMAT_VERSION)
        return Result<Header>::failure(INVALID, "unsupported interchange format version");
    h.kind = r.str();
    uint32_t nparams = r.u32();
    for (uint32_t i = 0; i < nparams && !r.bad; ++i) {
        std::string name = r.str();
        uint64_t v = r.u64();
        h.params[name] = v;
    }
    h.payload_len = r.u64();
    if (r.bad || h.payload_len > (uint64_t)(r.end - r.p))
        return Result<Header>::failure(INVALID, "truncated interchange blob");
    h.payload = r.p;
    h.next = r.p + h.payload_len;
    return Result<Header>::success(std::move(h));
}

void write_header(Writer &w, const std::string &kind, const std::map<std::string, uint64_t> &params, const std::vector<uint8_t> &payload) {
    w.out.insert(w.out.end(), MAGIC, MAGIC + 4);
    w.u32(FORMAT_VERSION);
    w.str(kind);
    w.u32((uint32_t)params.size());
    for (const auto &[k, v] : params) { w.str(k); w.u64(v); }
    w.u64(payload.size());
    w.bytes(payload);
}

Result<uint64_t> need(const Header &h, const char *name) {
    auto it = h.params.find(name);
    if (it == h.params.end()) return Result<uint64_t>::failure(INVALID, std::string("missing parameter ") + name + " in " + h.kind);
    return Result<uint64_t>::success(it->second);
}

Result<std::shared_ptr<Matrix>> decode_matrix(const Header &h) {
    auto p = need(h, "p"), count = need(h, "count"), rows = need(h, "rows"), cols = need(h, "cols");
    for (auto *r : {&p, &count, &rows, &cols})
        if (!r->ok) return Result<std::shared_ptr<Matrix>>::failure(r->error.status, r->error.message);
    if (!is_prime(p.value)) return Result<std::shared_ptr<Matrix>>::failure(INVALID, "p is not prime");
    if (p.value >= (1ULL << 32)) return Result<std::shared_ptr<Matrix>>::failure(INVALID, "p must be < 2^32");
    auto m = std::make_shared<Matrix>();
    m->p = p.value; m->count = count.value; m->rows = rows.value; m->cols = cols.value;
    unsigned w = entry_width(m->p);
    unsigned __int128 n = (unsigned __int128)m->count * m->rows * m->cols;
    if (n * w != h.payload_len) return Result<std::shared_ptr<Matrix>>::failure(INVALID, "gfp.matrix payload length does not match count*rows*cols");
    Reader r{h.payload, h.payload + h.payload_len};
    r.entries(m->entries, (uint64_t)n, w);
    for (Entry e : m->entries)
        if (e >= m->p) return Result<std::shared_ptr<Matrix>>::failure(INVALID, "entry not reduced mod p");
    return Result<std::shared_ptr<Matrix>>::success(m);
}

std::vector<uint8_t> encode_matrix(const Matrix &m) {
    Writer w;
    w.entries(m.entries, entry_width(m.p));
    Writer out;
    if (m.p == 0) write_header(out, "orbits.perms", {{"n", m.cols}, {"count", m.count}}, w.out);
    else if (m.p == NATURALS) write_header(out, "lk.naturals", {{"count", m.count}, {"rows", m.rows}, {"cols", m.cols}}, w.out);
    else write_header(out, "gfp.matrix", {{"p", m.p}, {"count", m.count}, {"rows", m.rows}, {"cols", m.cols}}, w.out);
    return out.out;
}

Result<std::shared_ptr<Matrix>> decode_naturals(const Header &h) {
    using R = Result<std::shared_ptr<Matrix>>;
    auto count = need(h, "count"), rows = need(h, "rows"), cols = need(h, "cols");
    for (auto *r : {&count, &rows, &cols}) if (!r->ok) return R::failure(r->error.status, r->error.message);
    if (rows.value == 0 || cols.value == 0) return R::failure(INVALID, "lk.naturals: need rows, cols >= 1");
    auto m = std::make_shared<Matrix>();
    m->p = NATURALS; m->count = count.value; m->rows = rows.value; m->cols = cols.value;
    unsigned __int128 total = (unsigned __int128)count.value * rows.value * cols.value;
    if (total * 4 != h.payload_len) return R::failure(INVALID, "lk.naturals payload length does not match count*rows*cols");
    Reader r{h.payload, h.payload + h.payload_len};
    r.entries(m->entries, (uint64_t)total, 4);
    return R::success(m);
}

Result<std::shared_ptr<Matrix>> decode_perms(const Header &h) {
    using R = Result<std::shared_ptr<Matrix>>;
    auto n = need(h, "n"), count = need(h, "count");
    for (auto *r : {&n, &count}) if (!r->ok) return R::failure(r->error.status, r->error.message);
    if (n.value == 0 || n.value >= (1ULL << 32)) return R::failure(INVALID, "orbits.perms: need 1 <= n < 2^32");
    auto m = std::make_shared<Matrix>();
    m->p = 0; m->count = count.value; m->rows = 1; m->cols = n.value;
    unsigned __int128 total = (unsigned __int128)count.value * n.value;
    if (total * 4 != h.payload_len) return R::failure(INVALID, "orbits.perms payload length does not match count*n");
    Reader r{h.payload, h.payload + h.payload_len};
    r.entries(m->entries, (uint64_t)total, 4);
    for (uint64_t g = 0; g < m->count; ++g) {
        std::vector<bool> seen(m->cols, false);
        for (uint64_t i = 0; i < m->cols; ++i) {
            Entry e = m->entries[g * m->cols + i];
            if (e >= m->cols || seen[e]) return R::failure(INVALID, "orbits.perms: element " + std::to_string(g) + " is not a permutation of 0..n-1");
            seen[e] = true;
        }
    }
    return R::success(m);
}

Result<std::shared_ptr<Family>> decode_family(const Header &h);

Result<std::shared_ptr<Object>> decode_at(const uint8_t *bytes, size_t len, const uint8_t **next);

Result<std::shared_ptr<Family>> decode_family(const Header &h) {
    using R = Result<std::shared_ptr<Family>>;
    std::string sub = h.kind.substr(7);
    const uint8_t *cur = h.payload, *end = h.payload + h.payload_len;
    auto child_object = [&](const char *expect) -> Result<std::shared_ptr<Object>> {
        const uint8_t *nx;
        auto o = decode_at(cur, (size_t)(end - cur), &nx);
        if (!o.ok) return o;
        bool matches = o.value->kind == expect ||
                       (std::string(expect) == "family" && o.value->kind.rfind("family.", 0) == 0) ||
                       (std::string(expect) == "matrix" && o.value->matrix);
        if (!matches)
            return Result<std::shared_ptr<Object>>::failure(INVALID, "family payload has the wrong kind");
        cur = nx;
        return o;
    };
    if (sub == "explicit") {
        auto b = child_object("gfp.matrix");
        if (!b.ok) return R::failure(b.error.status, b.error.message);
        return make_explicit(b.value->matrix);
    }
    if (sub == "subsets") {
        auto k = need(h, "k");
        if (!k.ok) return R::failure(k.error.status, k.error.message);
        auto d = child_object("matrix");
        if (!d.ok) return R::failure(d.error.status, d.error.message);
        return make_subsets(d.value->matrix, k.value);
    }
    if (sub == "subsets_of") {
        auto k = need(h, "k");
        if (!k.ok) return R::failure(k.error.status, k.error.message);
        auto inner = child_object("family");
        if (!inner.ok) return R::failure(inner.error.status, inner.error.message);
        return make_subsets_of(inner.value->family, k.value);
    }
    if (sub == "symmetric_matrices") {
        auto p = need(h, "p"), n = need(h, "n");
        for (auto *r : {&p, &n}) if (!r->ok) return R::failure(r->error.status, r->error.message);
        return make_symmetric_matrices(p.value, n.value);
    }
    if (sub == "range") {
        auto a = need(h, "a"), b = need(h, "b");
        for (auto *r : {&a, &b}) if (!r->ok) return R::failure(r->error.status, r->error.message);
        return make_range(a.value, b.value);
    }
    if (sub == "words") {
        auto q = need(h, "alphabet"), len = need(h, "length");
        for (auto *r : {&q, &len}) if (!r->ok) return R::failure(r->error.status, r->error.message);
        return make_words(q.value, len.value);
    }
    if (sub == "grassmannian") {
        auto p = need(h, "p"), n = need(h, "n"), hh = need(h, "h");
        for (auto *r : {&p, &n, &hh}) if (!r->ok) return R::failure(r->error.status, r->error.message);
        return make_grassmannian(p.value, n.value, hh.value);
    }
    if (sub == "group_elements") {
        auto g = child_object("orbits.perms");
        if (!g.ok) return R::failure(g.error.status, g.error.message);
        return make_group_elements(g.value->matrix);
    }
    if (sub == "all_matrices") {
        auto p = need(h, "p"), rows = need(h, "rows"), cols = need(h, "cols");
        for (auto *r : {&p, &rows, &cols}) if (!r->ok) return R::failure(r->error.status, r->error.message);
        return make_all_matrices(p.value, rows.value, cols.value);
    }
    if (sub == "transform" || sub == "stack") {
        auto inner = child_object("family");
        if (!inner.ok) return R::failure(inner.error.status, inner.error.message);
        auto mat = child_object("gfp.matrix");
        if (!mat.ok) return R::failure(mat.error.status, mat.error.message);
        return sub == "transform" ? make_transform(inner.value->family, mat.value->matrix)
                                  : make_stack(inner.value->family, mat.value->matrix);
    }
    return R::failure(INVALID, "unknown family kind " + h.kind);
}

std::vector<uint8_t> encode_family(const Family &f) {
    Writer payload;
    std::map<std::string, uint64_t> params;
    switch (f.kind) {
    case Family::Kind::Explicit:
    case Family::Kind::Subsets:
        payload.bytes(encode_matrix(*f.data));
        if (f.kind == Family::Kind::Subsets) params["k"] = f.k;
        break;
    case Family::Kind::Grassmannian:
        params = {{"p", f.p}, {"n", f.n}, {"h", f.h}};
        break;
    case Family::Kind::AllMatrices:
        params = {{"p", f.p}, {"rows", f.m}, {"cols", f.n}};
        break;
    case Family::Kind::SubsetsOf:
        payload.bytes(encode_family(*f.child));
        params["k"] = f.k;
        break;
    case Family::Kind::SymmetricMatrices:
        params = {{"p", f.p}, {"n", f.n}};
        break;
    case Family::Kind::Range:
        params = {{"a", f.a}, {"b", f.b}};
        break;
    case Family::Kind::Words:
        params = {{"alphabet", f.p}, {"length", f.n}};
        break;
    case Family::Kind::GroupElements:
        payload.bytes(encode_matrix(*f.data));
        break;
    case Family::Kind::Transform:
    case Family::Kind::Stack:
        payload.bytes(encode_family(*f.child));
        payload.bytes(encode_matrix(*f.data));
        break;
    }
    Writer out;
    write_header(out, std::string("family.") + family_kind_name(f.kind), params, payload.out);
    return out.out;
}

Result<std::shared_ptr<Object>> decode_at(const uint8_t *bytes, size_t len, const uint8_t **next) {
    using R = Result<std::shared_ptr<Object>>;
    auto hr = read_header(bytes, len);
    if (!hr.ok) return R::failure(hr.error.status, hr.error.message);
    const Header &h = hr.value;
    *next = h.next;
    auto o = std::make_shared<Object>();
    o->kind = h.kind;
    if (h.kind == "gfp.matrix" || h.kind == "orbits.perms" || h.kind == "lk.naturals") {
        auto m = h.kind == "gfp.matrix" ? decode_matrix(h) : h.kind == "orbits.perms" ? decode_perms(h) : decode_naturals(h);
        if (!m.ok) return R::failure(m.error.status, m.error.message);
        o->matrix = m.value;
        return R::success(o);
    }
    if (h.kind.rfind("family.", 0) == 0) {
        auto f = decode_family(h);
        if (!f.ok) return R::failure(f.error.status, f.error.message);
        o->family = f.value;
        return R::success(o);
    }
    if (h.kind == "count") {
        auto v = need(h, "value"), vis = need(h, "visited"), fs = need(h, "family_size");
        for (auto *r : {&v, &vis, &fs}) if (!r->ok) return R::failure(r->error.status, r->error.message);
        o->count = std::make_shared<Count>(Count{v.value, vis.value, fs.value});
        return R::success(o);
    }
    if (h.kind == "integers" || h.kind == "burnside.counts") {
        auto n = need(h, "count");
        if (!n.ok) return R::failure(n.error.status, n.error.message);
        if (n.value * 8 != h.payload_len) return R::failure(INVALID, h.kind + " payload length mismatch");
        Reader r{h.payload, h.payload + h.payload_len};
        o->integers = std::make_shared<Integers>();
        for (uint64_t i = 0; i < n.value; ++i) o->integers->values.push_back(r.u64());
        return R::success(o);
    }
    if (h.kind == "burnside.cycle_index") {
        auto degree = need(h, "degree"), count = need(h, "count"), denominator = need(h, "denominator");
        for (auto *r : {&degree, &count, &denominator}) if (!r->ok) return R::failure(r->error.status, r->error.message);
        unsigned __int128 words = (unsigned __int128)count.value * (degree.value + 1);
        if (words * 8 != h.payload_len) return R::failure(INVALID, "burnside.cycle_index payload length mismatch");
        auto index = std::make_shared<CycleIndex>();
        index->degree = degree.value;
        index->denominator = denominator.value;
        Reader r{h.payload, h.payload + h.payload_len};
        for (uint64_t i = 0; i < count.value; ++i) {
            index->multiplicities.push_back(r.u64());
            for (uint64_t j = 0; j < degree.value; ++j) index->cycles.push_back(r.u64());
        }
        o->cycle_index = index;
        return R::success(o);
    }
    if (h.kind == "histogram") {
        auto vis = need(h, "visited"), fs = need(h, "family_size"), bins = need(h, "bins");
        for (auto *r : {&vis, &fs, &bins}) if (!r->ok) return R::failure(r->error.status, r->error.message);
        if (bins.value * 8 != h.payload_len) return R::failure(INVALID, "histogram payload length mismatch");
        Reader r{h.payload, h.payload + h.payload_len};
        o->histogram = std::make_shared<Histogram>();
        o->histogram->visited = vis.value; o->histogram->family_size = fs.value;
        for (uint64_t i = 0; i < bins.value; ++i) o->histogram->bins.push_back(r.u64());
        return R::success(o);
    }
    if (h.kind == "hits") {
        auto p = need(h, "p"), rows = need(h, "rows"), cols = need(h, "cols"), total = need(h, "total"),
             vis = need(h, "visited"), fs = need(h, "family_size"), count = need(h, "count"), mat = need(h, "materialised");
        for (auto *r : {&p, &rows, &cols, &total, &vis, &fs, &count, &mat}) if (!r->ok) return R::failure(r->error.status, r->error.message);
        auto hh = std::make_shared<Hits>();
        hh->p = p.value; hh->rows = rows.value; hh->cols = cols.value; hh->total = total.value;
        hh->visited = vis.value; hh->family_size = fs.value;
        Reader r{h.payload, h.payload + h.payload_len};
        for (uint64_t i = 0; i < count.value; ++i) hh->indices.push_back(r.u64());
        r.entries(hh->members, mat.value * rows.value * cols.value, entry_width(p.value));
        if (r.bad || r.p != r.end) return R::failure(INVALID, "hits payload length mismatch");
        o->hits = hh;
        return R::success(o);
    }
    if (h.kind == "first") {
        auto p = need(h, "p"), rows = need(h, "rows"), cols = need(h, "cols"), found = need(h, "found"),
             index = need(h, "index"), vis = need(h, "visited"), fs = need(h, "family_size");
        for (auto *r : {&p, &rows, &cols, &found, &index, &vis, &fs}) if (!r->ok) return R::failure(r->error.status, r->error.message);
        auto f = std::make_shared<First>();
        f->p = p.value; f->rows = rows.value; f->cols = cols.value; f->found = found.value; f->index = index.value;
        f->visited = vis.value; f->family_size = fs.value;
        Reader r{h.payload, h.payload + h.payload_len};
        r.entries(f->member, found.value ? rows.value * cols.value : 0, entry_width(p.value));
        if (r.bad || r.p != r.end) return R::failure(INVALID, "first payload length mismatch");
        o->first = f;
        return R::success(o);
    }
    if (h.kind == "extremum") {
        auto p = need(h, "p"), rows = need(h, "rows"), cols = need(h, "cols"), value = need(h, "value"),
             index = need(h, "index"), vis = need(h, "visited"), fs = need(h, "family_size");
        for (auto *r : {&p, &rows, &cols, &value, &index, &vis, &fs}) if (!r->ok) return R::failure(r->error.status, r->error.message);
        auto e = std::make_shared<Extremum>();
        e->p = p.value; e->rows = rows.value; e->cols = cols.value; e->value = value.value; e->index = index.value;
        e->visited = vis.value; e->family_size = fs.value;
        Reader r{h.payload, h.payload + h.payload_len};
        r.entries(e->member, rows.value * cols.value, entry_width(p.value));
        if (r.bad || r.p != r.end) return R::failure(INVALID, "extremum payload length mismatch");
        o->extremum = e;
        return R::success(o);
    }
    if (h.kind == "gfp.basis") {
        auto p = need(h, "p"), count = need(h, "count"), cols = need(h, "cols");
        for (auto *r : {&p, &count, &cols}) if (!r->ok) return R::failure(r->error.status, r->error.message);
        auto b = std::make_shared<Basis>();
        b->p = p.value; b->count = count.value; b->cols = cols.value;
        Reader r{h.payload, h.payload + h.payload_len};
        for (uint64_t i = 0; i <= count.value; ++i) b->offsets.push_back(r.u64());
        if (r.bad) return R::failure(INVALID, "gfp.basis payload truncated");
        r.entries(b->entries, b->offsets.back() * cols.value, entry_width(p.value));
        if (r.bad || r.p != r.end) return R::failure(INVALID, "gfp.basis payload length mismatch");
        o->basis = b;
        return R::success(o);
    }
    if (h.kind == "gfp.solutions" || h.kind == "gfp.inverses") {
        bool sol = h.kind == "gfp.solutions";
        auto p = need(h, "p"), count = need(h, "count"), len = need(h, sol ? "length" : "n");
        for (auto *r : {&p, &count, &len}) if (!r->ok) return R::failure(r->error.status, r->error.message);
        uint64_t per = sol ? len.value : len.value * len.value;
        Reader r{h.payload, h.payload + h.payload_len};
        std::vector<uint8_t> flags(h.payload, h.payload + std::min<uint64_t>(count.value, h.payload_len));
        if (flags.size() != count.value) return R::failure(INVALID, "flags truncated");
        r.p += count.value;
        std::vector<Entry> e;
        r.entries(e, count.value * per, entry_width(p.value));
        if (r.bad || r.p != r.end) return R::failure(INVALID, h.kind + " payload length mismatch");
        if (sol) o->solutions = std::make_shared<Solutions>(Solutions{p.value, count.value, len.value, flags, e});
        else o->inverses = std::make_shared<Inverses>(Inverses{p.value, count.value, len.value, flags, e});
        return R::success(o);
    }
    if (h.kind == "gfp.witness") {
        auto p = need(h, "p"), count = need(h, "count"), rows = need(h, "rows"), cols = need(h, "cols");
        for (auto *r : {&p, &count, &rows, &cols}) if (!r->ok) return R::failure(r->error.status, r->error.message);
        auto w = std::make_shared<Witness>();
        w->p = p.value; w->count = count.value; w->rows = rows.value; w->cols = cols.value;
        Reader r{h.payload, h.payload + h.payload_len};
        r.entries(w->r, count.value * rows.value * cols.value, entry_width(p.value));
        r.entries(w->t, count.value * rows.value * rows.value, entry_width(p.value));
        if (r.bad || r.p != r.end) return R::failure(INVALID, "gfp.witness payload length mismatch");
        o->witness = w;
        return R::success(o);
    }
    return R::failure(INVALID, "unknown object kind " + h.kind);
}

} // namespace

unsigned entry_width(uint64_t p) {
    if (p == 0 || p == NATURALS) return 4; /* permutations: point indices; naturals */
    if (p < (1ULL << 8)) return 1;
    if (p < (1ULL << 16)) return 2;
    if (p < (1ULL << 32)) return 4;
    return 8;
}

bool is_prime(uint64_t n) {
    if (n < 2) return false;
    for (uint64_t q : {2ULL, 3ULL, 5ULL, 7ULL, 11ULL, 13ULL, 17ULL, 19ULL, 23ULL, 29ULL, 31ULL, 37ULL}) {
        if (n == q) return true;
        if (n % q == 0) return false;
    }
    uint64_t d = n - 1;
    int s = 0;
    while ((d & 1) == 0) { d >>= 1; ++s; }
    auto mulmod = [&](uint64_t a, uint64_t b) { return (uint64_t)((unsigned __int128)a * b % n); };
    auto powmod = [&](uint64_t a, uint64_t e) {
        uint64_t r = 1;
        while (e) { if (e & 1) r = mulmod(r, a); a = mulmod(a, a); e >>= 1; }
        return r;
    };
    for (uint64_t a : {2ULL, 3ULL, 5ULL, 7ULL, 11ULL, 13ULL, 17ULL, 19ULL, 23ULL, 29ULL, 31ULL, 37ULL}) {
        uint64_t x = powmod(a, d);
        if (x == 1 || x == n - 1) continue;
        bool comp = true;
        for (int i = 1; i < s; ++i) {
            x = mulmod(x, x);
            if (x == n - 1) { comp = false; break; }
        }
        if (comp) return false;
    }
    return true;
}

std::map<std::string, uint64_t> Object::params() const {
    if (matrix && matrix->p == 0) return {{"n", matrix->cols}, {"count", matrix->count}};
    if (matrix && matrix->p == NATURALS) return {{"count", matrix->count}, {"rows", matrix->rows}, {"cols", matrix->cols}};
    if (matrix) return {{"p", matrix->p}, {"count", matrix->count}, {"rows", matrix->rows}, {"cols", matrix->cols}};
    if (basis) return {{"p", basis->p}, {"count", basis->count}, {"cols", basis->cols}};
    if (solutions) return {{"p", solutions->p}, {"count", solutions->count}, {"length", solutions->length}};
    if (inverses) return {{"p", inverses->p}, {"count", inverses->count}, {"n", inverses->n}};
    if (witness) return {{"p", witness->p}, {"count", witness->count}, {"rows", witness->rows}, {"cols", witness->cols}};
    if (cycle_index) return {{"degree", cycle_index->degree}, {"count", cycle_index->multiplicities.size()},
                             {"denominator", cycle_index->denominator}};
    if (integers) return {{"count", integers->values.size()}};
    if (count) return {{"value", count->value}, {"visited", count->visited}, {"family_size", count->family_size}};
    if (histogram) return {{"visited", histogram->visited}, {"family_size", histogram->family_size}, {"bins", histogram->bins.size()}};
    if (hits) return {{"p", hits->p}, {"rows", hits->rows}, {"cols", hits->cols}, {"total", hits->total},
                      {"visited", hits->visited}, {"family_size", hits->family_size}, {"count", hits->indices.size()},
                      {"materialised", (hits->rows * hits->cols) != 0 ? hits->members.size() / (hits->rows * hits->cols) : 0}};
    if (first) return {{"p", first->p}, {"rows", first->rows}, {"cols", first->cols}, {"found", first->found}, {"index", first->index},
                       {"visited", first->visited}, {"family_size", first->family_size}};
    if (extremum) return {{"p", extremum->p}, {"rows", extremum->rows}, {"cols", extremum->cols}, {"value", extremum->value},
                          {"index", extremum->index}, {"visited", extremum->visited}, {"family_size", extremum->family_size}};
    if (family) {
        std::map<std::string, uint64_t> m{{"p", family->prime()}, {"rows", family->rows()}, {"cols", family->cols()}};
        auto sz = family->size();
        if (sz.ok) m["size"] = sz.value;
        return m;
    }
    return {};
}

Result<std::shared_ptr<Object>> decode(const uint8_t *bytes, size_t len) {
    const uint8_t *next;
    auto r = decode_at(bytes, len, &next);
    if (r.ok && next != bytes + len) return Result<std::shared_ptr<Object>>::failure(INVALID, "trailing bytes after interchange blob");
    return r;
}

std::vector<uint8_t> encode(const Object &o) {
    Writer w;
    if (o.matrix) return encode_matrix(*o.matrix);
    if (o.family) return encode_family(*o.family);
    Writer out;
    if (o.basis) {
        w.u64s(o.basis->offsets);
        w.entries(o.basis->entries, entry_width(o.basis->p));
        write_header(out, "gfp.basis", o.params(), w.out);
    } else if (o.solutions) {
        w.bytes(o.solutions->solvable);
        w.entries(o.solutions->entries, entry_width(o.solutions->p));
        write_header(out, "gfp.solutions", o.params(), w.out);
    } else if (o.inverses) {
        w.bytes(o.inverses->invertible);
        w.entries(o.inverses->entries, entry_width(o.inverses->p));
        write_header(out, "gfp.inverses", o.params(), w.out);
    } else if (o.witness) {
        w.entries(o.witness->r, entry_width(o.witness->p));
        w.entries(o.witness->t, entry_width(o.witness->p));
        write_header(out, "gfp.witness", o.params(), w.out);
    } else if (o.cycle_index) {
        for (uint64_t i = 0; i < o.cycle_index->multiplicities.size(); ++i) {
            w.u64(o.cycle_index->multiplicities[i]);
            for (uint64_t j = 0; j < o.cycle_index->degree; ++j)
                w.u64(o.cycle_index->cycles[i * o.cycle_index->degree + j]);
        }
        write_header(out, "burnside.cycle_index", o.params(), w.out);
    } else if (o.integers) {
        w.u64s(o.integers->values);
        write_header(out, o.kind == "burnside.counts" ? "burnside.counts" : "integers", o.params(), w.out);
    } else if (o.count) {
        write_header(out, "count", o.params(), {});
    } else if (o.histogram) {
        w.u64s(o.histogram->bins);
        write_header(out, "histogram", o.params(), w.out);
    } else if (o.hits) {
        w.u64s(o.hits->indices);
        w.entries(o.hits->members, entry_width(o.hits->p));
        write_header(out, "hits", o.params(), w.out);
    } else if (o.first) {
        w.entries(o.first->member, entry_width(o.first->p));
        write_header(out, "first", o.params(), w.out);
    } else if (o.extremum) {
        w.entries(o.extremum->member, entry_width(o.extremum->p));
        write_header(out, "extremum", o.params(), w.out);
    }
    return out.out;
}

} // namespace lk
