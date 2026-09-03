#include "object.hpp"
#include "family.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <stdexcept>

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
        if (width && e.size() > (out.max_size() - start) / width)
            throw std::length_error("interchange object is too large");
        out.resize(start + e.size() * width);
        uint8_t *dst = out.data() + start;
        for (Entry v : e) {
            for (unsigned i = 0; i < width; ++i) *dst++ = (uint8_t)((uint64_t)v >> (8 * i));
        }
    }
    void u64s(const std::vector<uint64_t> &v) { for (uint64_t x : v) u64(x); }
    void i64s(const std::vector<int64_t> &v) { for (int64_t x : v) u64((uint64_t)x); }
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
    int64_t i64() { return std::bit_cast<int64_t>(u64()); }
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
    if (p.value < 2 || p.value >= (1ULL << 32))
        return Result<std::shared_ptr<Matrix>>::failure(INVALID, "field-size tag must satisfy 2 <= p < 2^32");
    auto m = std::make_shared<Matrix>();
    m->p = p.value; m->count = count.value; m->rows = rows.value; m->cols = cols.value;
    unsigned w = entry_width(m->p);
    unsigned __int128 n = (unsigned __int128)m->count * m->rows * m->cols;
    if (n * w != h.payload_len) return Result<std::shared_ptr<Matrix>>::failure(INVALID, "gfp.matrix payload length does not match count*rows*cols");
    Reader r{h.payload, h.payload + h.payload_len};
    r.entries(m->entries, (uint64_t)n, w);
    for (Entry e : m->entries)
        if (e >= m->p) return Result<std::shared_ptr<Matrix>>::failure(INVALID, "matrix entry is outside 0..p-1");
    return Result<std::shared_ptr<Matrix>>::success(m);
}

std::vector<uint8_t> encode_matrix(const Matrix &m) {
    Writer w;
    w.entries(m.entries, entry_width(m.p));
    Writer out;
    if (m.p == 0) write_header(out, "orbits.perms", {{"n", m.cols}, {"count", m.count * m.rows}}, w.out);
    else if (m.p == NATURALS) write_header(out, "lk.naturals", {{"count", m.count}, {"rows", m.rows}, {"cols", m.cols}}, w.out);
    else if (m.p == GRAMS) write_header(out, "lattices.gram", {{"count", m.count}, {"n", m.rows}}, w.out);
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

Result<std::shared_ptr<Matrix>> decode_gram(const Header &h) {
    using R = Result<std::shared_ptr<Matrix>>;
    auto count = need(h, "count"), n = need(h, "n");
    for (auto *r : {&count, &n}) if (!r->ok) return R::failure(r->error.status, r->error.message);
    if (n.value == 0) return R::failure(INVALID, "lattices.gram: need n >= 1");
    unsigned __int128 total = (unsigned __int128)count.value * n.value * n.value;
    if (total * 4 != h.payload_len) return R::failure(INVALID, "lattices.gram payload length does not match count*n*n");
    auto m = std::make_shared<Matrix>();
    m->p = GRAMS; m->count = count.value; m->rows = n.value; m->cols = n.value;
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
        /* any matrix batch: residues, permutations or naturals, as make_explicit accepts */
        auto b = child_object("matrix");
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
    if (sub == "symmetric_matrices" || sub == "alternating_matrices") {
        auto p = need(h, "p"), n = need(h, "n");
        for (auto *r : {&p, &n}) if (!r->ok) return R::failure(r->error.status, r->error.message);
        return sub == "symmetric_matrices" ? make_symmetric_matrices(p.value, n.value)
                                            : make_alternating_matrices(p.value, n.value);
    }
    if (sub == "all_graphs") {
        auto n = need(h, "n");
        if (!n.ok) return R::failure(n.error.status, n.error.message);
        return make_all_graphs(n.value);
    }
    if (sub == "edge_subgraphs") {
        auto k = need(h, "k");
        if (!k.ok) return R::failure(k.error.status, k.error.message);
        auto host = child_object("gfp.matrix");
        if (!host.ok) return R::failure(host.error.status, host.error.message);
        return make_edge_subgraphs(host.value->matrix, k.value);
    }
    if (sub == "cayley_graphs") {
        auto group = child_object("orbits.perms");
        if (!group.ok) return R::failure(group.error.status, group.error.message);
        return make_cayley_graphs(group.value->matrix);
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
    if (sub == "latin_squares") {
        auto n = need(h, "n");
        if (!n.ok) return R::failure(n.error.status, n.error.message);
        return make_latin_squares(n.value);
    }
    if (sub == "partitions") {
        auto total = need(h, "total"), max_part = need(h, "max_part"), max_parts = need(h, "max_parts"),
             max_multiplicity = need(h, "max_multiplicity"), distinct = need(h, "distinct"), odd = need(h, "odd");
        for (auto *r : {&total, &max_part, &max_parts, &max_multiplicity, &distinct, &odd})
            if (!r->ok) return R::failure(r->error.status, r->error.message);
        return make_partitions(total.value, max_part.value, max_parts.value, max_multiplicity.value,
                               distinct.value, odd.value);
    }
    if (sub == "compositions") {
        auto total = need(h, "total"), parts = need(h, "parts"), max_part = need(h, "max_part");
        for (auto *r : {&total, &parts, &max_part}) if (!r->ok) return R::failure(r->error.status, r->error.message);
        return make_compositions(total.value, parts.value, max_part.value);
    }
    if (sub == "standard_tableaux") {
        auto shape = child_object("matrix");
        if (!shape.ok) return R::failure(shape.error.status, shape.error.message);
        return make_standard_tableaux(shape.value->matrix);
    }
    if (sub == "sublattices") {
        auto index = need(h, "index");
        if (!index.ok) return R::failure(index.error.status, index.error.message);
        auto gram = child_object("lattices.gram");
        if (!gram.ok) return R::failure(gram.error.status, gram.error.message);
        return make_sublattices(gram.value->matrix, index.value);
    }
    if (sub == "grassmannian") {
        auto p = need(h, "p"), n = need(h, "n"), hh = need(h, "h");
        for (auto *r : {&p, &n, &hh}) if (!r->ok) return R::failure(r->error.status, r->error.message);
        return make_grassmannian(p.value, n.value, hh.value);
    }
    if (sub == "group_elements") {
        auto g = child_object("matrix");
        if (!g.ok) return R::failure(g.error.status, g.error.message);
        return make_group_elements(g.value->matrix);
    }
    if (sub == "group_tables") {
        auto tables = child_object("lk.naturals");
        if (!tables.ok) return R::failure(tables.error.status, tables.error.message);
        return make_group_tables(tables.value->matrix);
    }
    if (sub == "all_matrices") {
        auto p = need(h, "p"), rows = need(h, "rows"), cols = need(h, "cols");
        for (auto *r : {&p, &rows, &cols}) if (!r->ok) return R::failure(r->error.status, r->error.message);
        return make_all_matrices(p.value, rows.value, cols.value);
    }
    if (sub == "transform" || sub == "stack") {
        auto inner = child_object("family");
        if (!inner.ok) return R::failure(inner.error.status, inner.error.message);
        /* transform multiplies, so it needs residues; stack only appends rows, of any kind */
        auto mat = child_object(sub == "stack" ? "matrix" : "gfp.matrix");
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
    case Family::Kind::AlternatingMatrices:
        params = {{"p", f.p}, {"n", f.n}};
        break;
    case Family::Kind::Range:
        params = {{"a", f.a}, {"b", f.b}};
        break;
    case Family::Kind::Words:
        params = {{"alphabet", f.p}, {"length", f.n}};
        break;
    case Family::Kind::LatinSquares:
        params = {{"n", f.n}};
        break;
    case Family::Kind::Partitions:
        params = {{"total", f.n}, {"max_part", f.m}, {"max_parts", f.k}, {"max_multiplicity", f.h},
                  {"distinct", f.a}, {"odd", f.b}};
        break;
    case Family::Kind::Compositions:
        params = {{"total", f.n}, {"parts", f.k}, {"max_part", f.m}};
        break;
    case Family::Kind::StandardTableaux:
        payload.bytes(encode_matrix(*f.data));
        break;
    case Family::Kind::AllGraphs:
        params = {{"n", f.n}};
        break;
    case Family::Kind::EdgeSubgraphs:
        payload.bytes(encode_matrix(*f.data));
        params = {{"k", f.k}};
        break;
    case Family::Kind::CayleyGraphs:
        payload.bytes(encode_matrix(*f.data));
        break;
    case Family::Kind::GroupElements:
    case Family::Kind::GroupTables:
        payload.bytes(encode_matrix(*f.data));
        break;
    case Family::Kind::Sublattices:
        payload.bytes(encode_matrix(*f.data));
        params["index"] = f.k;
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
    if (h.kind == "gfp.matrix" || h.kind == "orbits.perms" || h.kind == "lk.naturals" || h.kind == "lattices.gram") {
        auto m = h.kind == "gfp.matrix" ? decode_matrix(h) : h.kind == "orbits.perms" ? decode_perms(h) :
                 h.kind == "lk.naturals" ? decode_naturals(h) : decode_gram(h);
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
    if (h.kind == "integers" || h.kind == "burnside.counts" || h.kind == "characters.multiplicities") {
        auto n = need(h, "count");
        if (!n.ok) return R::failure(n.error.status, n.error.message);
        if (n.value * 8 != h.payload_len) return R::failure(INVALID, h.kind + " payload length mismatch");
        Reader r{h.payload, h.payload + h.payload_len};
        o->integers = std::make_shared<Integers>();
        for (uint64_t i = 0; i < n.value; ++i) o->integers->values.push_back(r.u64());
        return R::success(o);
    }
    if (h.kind == "elliptic_curves_fp.group") {
        auto n = need(h, "count");
        if (!n.ok) return R::failure(n.error.status, n.error.message);
        if (n.value * 16 != h.payload_len) return R::failure(INVALID, "elliptic_curves_fp.group payload length mismatch");
        Reader r{h.payload, h.payload + h.payload_len};
        o->curve_groups = std::make_shared<CurveGroups>();
        o->curve_groups->count = n.value;
        for (uint64_t i = 0; i < 2 * n.value; ++i) o->curve_groups->orders.push_back(r.u64());
        return R::success(o);
    }
    if (h.kind == "graphs.degree_sequences") {
        auto count = need(h, "count"), n = need(h, "n");
        for (auto *r : {&count, &n}) if (!r->ok) return R::failure(r->error.status, r->error.message);
        if ((unsigned __int128)count.value * n.value * 4 != h.payload_len)
            return R::failure(INVALID, "graphs.degree_sequences payload length mismatch");
        auto d = std::make_shared<DegreeSequences>();
        d->count = count.value;
        d->n = n.value;
        Reader r{h.payload, h.payload + h.payload_len};
        r.entries(d->entries, count.value * n.value, 4);
        o->degree_sequences = d;
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
    if (h.kind == "characters.table") {
        auto order = need(h, "order"), classes = need(h, "classes"), conductor = need(h, "conductor");
        for (auto *r : {&order, &classes, &conductor}) if (!r->ok) return R::failure(r->error.status, r->error.message);
        if (order.value == 0 || classes.value == 0 || conductor.value == 0 || conductor.value >= (1ULL << 32))
            return R::failure(INVALID, "characters.table needs positive order, classes, and a conductor below 2^32");
        unsigned __int128 header_bytes = (unsigned __int128)classes.value * 3 * 8;
        if (header_bytes > h.payload_len) return R::failure(INVALID, "characters.table payload truncated");
        auto table = std::make_shared<CharacterTable>();
        table->order = order.value; table->classes = classes.value; table->conductor = conductor.value;
        Reader r{h.payload, h.payload + h.payload_len};
        for (uint64_t i = 0; i < classes.value; ++i) table->representatives.push_back(r.u64());
        for (uint64_t i = 0; i < classes.value; ++i) table->class_sizes.push_back(r.u64());
        unsigned __int128 degree_sum = 0;
        for (uint64_t i = 0; i < classes.value; ++i) {
            uint64_t degree = r.u64();
            if (degree == 0) return R::failure(INVALID, "characters.table character degree is zero");
            table->degrees.push_back(degree);
            degree_sum += degree;
        }
        unsigned __int128 entries = degree_sum * classes.value;
        if (header_bytes + entries * 4 != h.payload_len || entries > UINT64_MAX)
            return R::failure(INVALID, "characters.table payload length mismatch");
        r.entries(table->spectra, (uint64_t)entries, 4);
        if (r.bad || r.p != r.end) return R::failure(INVALID, "characters.table payload length mismatch");
        uint64_t class_total = 0;
        for (uint64_t i = 0; i < classes.value; ++i) {
            if (table->representatives[i] >= order.value) return R::failure(INVALID, "characters.table representative index is outside the group");
            if (UINT64_MAX - class_total < table->class_sizes[i]) return R::failure(INVALID, "characters.table class sizes overflow");
            class_total += table->class_sizes[i];
        }
        if (class_total != order.value) return R::failure(INVALID, "characters.table class sizes do not sum to the group order");
        for (Entry exponent : table->spectra)
            if (exponent >= conductor.value) return R::failure(INVALID, "characters.table exponent is not reduced modulo the conductor");
        o->character_table = std::move(table);
        return R::success(o);
    }
    if (h.kind == "characters.indicators") {
        auto count = need(h, "count");
        if (!count.ok) return R::failure(count.error.status, count.error.message);
        if (count.value != h.payload_len) return R::failure(INVALID, "characters.indicators payload length mismatch");
        auto indicators = std::make_shared<CharacterIndicators>();
        indicators->values.reserve(count.value);
        for (uint64_t i = 0; i < count.value; ++i) {
            int8_t value = static_cast<int8_t>(h.payload[i]);
            if (value < -1 || value > 1) return R::failure(INVALID, "characters.indicators value is not -1, 0, or 1");
            indicators->values.push_back(value);
        }
        o->character_indicators = std::move(indicators);
        return R::success(o);
    }
    if (h.kind == "graph_polynomials.coefficients") {
        auto count = need(h, "count"), length = need(h, "length");
        for (auto *r : {&count, &length}) if (!r->ok) return R::failure(r->error.status, r->error.message);
        unsigned __int128 words = (unsigned __int128)count.value * length.value;
        if (words * 8 != h.payload_len) return R::failure(INVALID, "graph_polynomials.coefficients payload length mismatch");
        auto coefficients = std::make_shared<Coefficients>();
        coefficients->count = count.value;
        coefficients->length = length.value;
        Reader r{h.payload, h.payload + h.payload_len};
        for (uint64_t i = 0; i < (uint64_t)words; ++i) coefficients->values.push_back(r.i64());
        o->coefficients = coefficients;
        return R::success(o);
    }
    if (h.kind == "circulants.spectra") {
        auto n = need(h, "n"), count = need(h, "count");
        for (auto *r : {&n, &count}) if (!r->ok) return R::failure(r->error.status, r->error.message);
        if (n.value == 0 || n.value >= (1ULL << 32))
            return R::failure(INVALID, "circulants.spectra: need 1 <= n < 2^32");
        unsigned __int128 rows = (unsigned __int128)count.value * n.value;
        unsigned __int128 offset_bytes = (rows + 1) * 8;
        if (offset_bytes > h.payload_len || rows + 1 > SIZE_MAX / sizeof(uint64_t))
            return R::failure(INVALID, "circulants.spectra payload length mismatch");
        auto spectra = std::make_shared<Spectra>();
        spectra->n = n.value; spectra->count = count.value;
        Reader r{h.payload, h.payload + h.payload_len};
        spectra->offsets.resize((size_t)(rows + 1));
        for (uint64_t i = 0; i <= (uint64_t)rows; ++i) spectra->offsets[i] = r.u64();
        if (r.bad || spectra->offsets.front() != 0 ||
            !std::is_sorted(spectra->offsets.begin(), spectra->offsets.end()) ||
            (unsigned __int128)spectra->offsets.back() * 4 != (uint64_t)(r.end - r.p))
            return R::failure(INVALID, "circulants.spectra payload length mismatch");
        r.entries(spectra->exponents, spectra->offsets.back(), 4);
        if (r.bad || r.p != r.end) return R::failure(INVALID, "circulants.spectra payload length mismatch");
        for (uint64_t row = 0; row < (uint64_t)rows; ++row) {
            auto first = spectra->exponents.begin() + spectra->offsets[row];
            auto last = spectra->exponents.begin() + spectra->offsets[row + 1];
            if (!std::is_sorted(first, last) || std::any_of(first, last, [&](Entry e) { return e >= n.value; }))
                return R::failure(INVALID, "circulants.spectra rows must contain sorted exponents below n");
        }
        o->spectra = spectra;
        return R::success(o);
    }
    if (h.kind == "linear_codes.weight_enumerators") {
        auto count = need(h, "count"), n = need(h, "n");
        for (auto *r : {&count, &n}) if (!r->ok) return R::failure(r->error.status, r->error.message);
        unsigned __int128 words = (unsigned __int128)count.value * (n.value + 1);
        if (words * 8 != h.payload_len) return R::failure(INVALID, "linear_codes.weight_enumerators payload length mismatch");
        auto enumerators = std::make_shared<WeightEnumerators>();
        enumerators->count = count.value;
        enumerators->n = n.value;
        Reader r{h.payload, h.payload + h.payload_len};
        for (unsigned __int128 i = 0; i < words; ++i) enumerators->coefficients.push_back(r.u64());
        o->weight_enumerators = enumerators;
        return R::success(o);
    }
    if (h.kind == "lattices.theta_series") {
        auto count = need(h, "count"), bound = need(h, "bound");
        for (auto *r : {&count, &bound}) if (!r->ok) return R::failure(r->error.status, r->error.message);
        unsigned __int128 words = (unsigned __int128)count.value * ((unsigned __int128)bound.value + 1);
        if (words * 8 != h.payload_len) return R::failure(INVALID, "lattices.theta_series payload length mismatch");
        o->theta_series = std::make_shared<ThetaSeries>();
        o->theta_series->count = count.value; o->theta_series->bound = bound.value;
        Reader r{h.payload, h.payload + h.payload_len};
        for (uint64_t i = 0; i < (uint64_t)words; ++i) o->theta_series->coefficients.push_back(r.u64());
        return R::success(o);
    }
    if (h.kind == "lattices.short_vectors") {
        auto count = need(h, "count"), n = need(h, "n"), bound = need(h, "bound");
        for (auto *r : {&count, &n, &bound}) if (!r->ok) return R::failure(r->error.status, r->error.message);
        if (n.value == 0) return R::failure(INVALID, "lattices.short_vectors: need n >= 1");
        unsigned __int128 offset_words = (unsigned __int128)count.value + 1;
        if (offset_words * 8 > h.payload_len)
            return R::failure(INVALID, "lattices.short_vectors offsets truncated");
        auto sv = std::make_shared<ShortVectors>();
        sv->count = count.value; sv->n = n.value; sv->bound = bound.value;
        Reader r{h.payload, h.payload + h.payload_len};
        for (uint64_t i = 0; i < count.value + 1; ++i) sv->offsets.push_back(r.u64());
        if (r.bad || sv->offsets[0] != 0)
            return R::failure(INVALID, "lattices.short_vectors offsets must start at zero");
        for (uint64_t i = 1; i < sv->offsets.size(); ++i)
            if (sv->offsets[i] < sv->offsets[i - 1])
                return R::failure(INVALID, "lattices.short_vectors offsets must be nondecreasing");
        unsigned __int128 entries = (unsigned __int128)sv->offsets.back() * n.value;
        if (entries * 4 != (uint64_t)(r.end - r.p))
            return R::failure(INVALID, "lattices.short_vectors payload length mismatch");
        r.entries(sv->entries, (uint64_t)entries, 4);
        if (r.bad || r.p != r.end) return R::failure(INVALID, "lattices.short_vectors payload length mismatch");
        o->short_vectors = sv;
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
    if (h.kind == "young.characters") {
        auto count = need(h, "count");
        if (!count.ok) return R::failure(count.error.status, count.error.message);
        if (count.value * 8 != h.payload_len) return R::failure(INVALID, "young.characters payload length mismatch");
        Reader r{h.payload, h.payload + h.payload_len};
        o->characters = std::make_shared<Characters>();
        for (uint64_t i = 0; i < count.value; ++i) o->characters->values.push_back(r.i64());
        return R::success(o);
    }
    if (h.kind == "young.rsk_pairs") {
        auto count = need(h, "count"), length = need(h, "length");
        for (auto *r : {&count, &length}) if (!r->ok) return R::failure(r->error.status, r->error.message);
        unsigned __int128 entries = (unsigned __int128)count.value * length.value *
                                    (1 + 2 * (unsigned __int128)length.value);
        if (entries * 4 != h.payload_len) return R::failure(INVALID, "young.rsk_pairs payload length mismatch");
        auto pairs = std::make_shared<RskPairs>();
        pairs->count = count.value; pairs->length = length.value;
        Reader r{h.payload, h.payload + h.payload_len};
        r.entries(pairs->shapes, count.value * length.value, 4);
        r.entries(pairs->insertion, count.value * length.value * length.value, 4);
        r.entries(pairs->recording, count.value * length.value * length.value, 4);
        if (r.bad || r.p != r.end) return R::failure(INVALID, "young.rsk_pairs payload length mismatch");
        o->rsk_pairs = pairs;
        return R::success(o);
    }
    if (h.kind == "graph_iso.groups") {
        using G = Result<std::shared_ptr<Object>>;
        auto count = need(h, "count"), n = need(h, "n");
        for (auto *r : {&count, &n}) if (!r->ok) return G::failure(r->error.status, r->error.message);
        if (n.value == 0 || n.value >= (1ULL << 32))
            return G::failure(INVALID, "graph_iso.groups: need 1 <= n < 2^32");
        if (((unsigned __int128)count.value + 1) * 8 > h.payload_len)
            return G::failure(INVALID, "graph_iso.groups offsets truncated");
        auto groups = std::make_shared<GraphGroups>();
        groups->count = count.value; groups->n = n.value;
        Reader r{h.payload, h.payload + h.payload_len};
        for (uint64_t i = 0; i <= count.value; ++i) groups->offsets.push_back(r.u64());
        if (r.bad || groups->offsets.front() != 0)
            return G::failure(INVALID, "graph_iso.groups offsets must start at zero");
        for (uint64_t i = 0; i < count.value; ++i)
            if (groups->offsets[i] > groups->offsets[i + 1])
                return G::failure(INVALID, "graph_iso.groups offsets must be nondecreasing");
        if (groups->offsets.back() > UINT64_MAX / n.value)
            return G::failure(INVALID, "graph_iso.groups entry count overflows u64");
        r.entries(groups->entries, groups->offsets.back() * n.value, 4);
        if (r.bad || r.p != r.end) return G::failure(INVALID, "graph_iso.groups payload length mismatch");
        for (uint64_t g = 0; g < groups->offsets.back(); ++g) {
            std::vector<bool> seen(n.value, false);
            for (uint64_t i = 0; i < n.value; ++i) {
                Entry e = groups->entries[g * n.value + i];
                if (e >= n.value || seen[e])
                    return G::failure(INVALID, "graph_iso.groups element " + std::to_string(g) + " is not a permutation of 0..n-1");
                seen[e] = true;
            }
        }
        o->graph_groups = groups;
        return G::success(o);
    }
    if (h.kind == "vertex_transitive.regular_subgroups") {
        auto count = need(h, "count"), n = need(h, "n");
        for (auto *r : {&count, &n}) if (!r->ok) return R::failure(r->error.status, r->error.message);
        if (n.value == 0 || n.value >= (1ULL << 32))
            return R::failure(INVALID, "vertex_transitive.regular_subgroups: need 1 <= n < 2^32");
        if (((unsigned __int128)count.value + 1) * 8 > h.payload_len)
            return R::failure(INVALID, "vertex_transitive.regular_subgroups offsets truncated");
        auto groups = std::make_shared<RegularSubgroups>();
        groups->count = count.value; groups->n = n.value;
        Reader r{h.payload, h.payload + h.payload_len};
        for (uint64_t i = 0; i <= count.value; ++i) groups->offsets.push_back(r.u64());
        if (r.bad || groups->offsets.front() != 0)
            return R::failure(INVALID, "vertex_transitive.regular_subgroups offsets must start at zero");
        for (uint64_t i = 0; i < count.value; ++i)
            if (groups->offsets[i] > groups->offsets[i + 1])
                return R::failure(INVALID, "vertex_transitive.regular_subgroups offsets must be nondecreasing");
        unsigned __int128 entry_count = (unsigned __int128)groups->offsets.back() * n.value * n.value;
        if (entry_count > UINT64_MAX)
            return R::failure(INVALID, "vertex_transitive.regular_subgroups entry count overflows u64");
        r.entries(groups->entries, (uint64_t)entry_count, 4);
        if (r.bad || r.p != r.end)
            return R::failure(INVALID, "vertex_transitive.regular_subgroups payload length mismatch");
        uint64_t permutations = groups->offsets.back() * n.value;
        for (uint64_t g = 0; g < permutations; ++g) {
            std::vector<bool> seen(n.value, false);
            for (uint64_t i = 0; i < n.value; ++i) {
                Entry e = groups->entries[g * n.value + i];
                if (e >= n.value || seen[e])
                    return R::failure(INVALID, "vertex_transitive.regular_subgroups contains an invalid permutation");
                seen[e] = true;
            }
        }
        o->regular_subgroups = groups;
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
    if (h.kind == "polynomials_fq.elements") {
        auto p = need(h, "p"), count = need(h, "count");
        for (auto *r : {&p, &count}) if (!r->ok) return R::failure(r->error.status, r->error.message);
        auto e = std::make_shared<Elements>();
        e->p = p.value; e->count = count.value;
        Reader r{h.payload, h.payload + h.payload_len};
        for (uint64_t i = 0; i <= count.value; ++i) e->offsets.push_back(r.u64());
        if (r.bad) return R::failure(INVALID, "polynomials_fq.elements payload truncated");
        r.entries(e->values, e->offsets.back(), entry_width(p.value));
        if (r.bad || r.p != r.end) return R::failure(INVALID, "polynomials_fq.elements payload length mismatch");
        o->elements = e;
        return R::success(o);
    }
    if (h.kind == "polynomials_fq.degrees") {
        auto count = need(h, "count");
        if (!count.ok) return R::failure(count.error.status, count.error.message);
        auto d = std::make_shared<Degrees>();
        d->count = count.value;
        Reader r{h.payload, h.payload + h.payload_len};
        for (uint64_t i = 0; i <= count.value; ++i) d->offsets.push_back(r.u64());
        if (r.bad) return R::failure(INVALID, "polynomials_fq.degrees payload truncated");
        for (uint64_t i = 0; i < d->offsets.back(); ++i) d->values.push_back(r.u64());
        if (r.bad || r.p != r.end) return R::failure(INVALID, "polynomials_fq.degrees payload length mismatch");
        o->degrees = d;
        return R::success(o);
    }
    if (h.kind == "continued_fractions_and_pell.expansion") {
        auto count = need(h, "count");
        if (!count.ok) return R::failure(count.error.status, count.error.message);
        auto e = std::make_shared<ContinuedFractions>();
        e->count = count.value;
        Reader r{h.payload, h.payload + h.payload_len};
        for (uint64_t i = 0; i <= count.value; ++i) e->offsets.push_back(r.u64());
        if (r.bad) return R::failure(INVALID, "continued_fractions_and_pell.expansion payload truncated");
        for (uint64_t i = 0; i < e->offsets.back(); ++i) e->values.push_back(r.u64());
        if (r.bad || r.p != r.end)
            return R::failure(INVALID, "continued_fractions_and_pell.expansion payload length mismatch");
        o->continued_fractions = e;
        return R::success(o);
    }
    if (h.kind == "continued_fractions_and_pell.unit") {
        auto count = need(h, "count");
        if (!count.ok) return R::failure(count.error.status, count.error.message);
        unsigned __int128 want = (unsigned __int128)count.value * 18; /* two flags and two u64 */
        if (want != h.payload_len)
            return R::failure(INVALID, "continued_fractions_and_pell.unit payload length mismatch");
        auto u = std::make_shared<QuadraticUnits>();
        u->count = count.value;
        u->solvable.assign(h.payload, h.payload + count.value);
        u->negative.assign(h.payload + count.value, h.payload + 2 * count.value);
        Reader r{h.payload + 2 * count.value, h.payload + h.payload_len};
        for (uint64_t i = 0; i < 2 * count.value; ++i) u->pairs.push_back(r.u64());
        if (r.bad || r.p != r.end)
            return R::failure(INVALID, "continued_fractions_and_pell.unit payload length mismatch");
        o->quadratic_units = u;
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
    if (h.kind == "sieve_ranges.factorisation") {
        auto count = need(h, "count");
        if (!count.ok) return R::failure(count.error.status, count.error.message);
        auto f = std::make_shared<Factorisation>();
        f->count = count.value;
        Reader r{h.payload, h.payload + h.payload_len};
        for (uint64_t i = 0; i <= count.value; ++i) f->offsets.push_back(r.u64());
        if (r.bad) return R::failure(INVALID, "sieve_ranges.factorisation payload truncated");
        for (uint64_t i = 1; i <= count.value; ++i)
            if (f->offsets[i] < f->offsets[i - 1]) return R::failure(INVALID, "sieve_ranges.factorisation offsets decrease");
        for (uint64_t i = 0; i < 2 * f->offsets.back(); ++i) f->pairs.push_back(r.u64());
        if (r.bad || r.p != r.end) return R::failure(INVALID, "sieve_ranges.factorisation payload length mismatch");
        o->factorisation = f;
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
    if (h.kind == "designs.matrix") {
        auto count = need(h, "count"), rows = need(h, "rows"), cols = need(h, "cols");
        for (auto *r : {&count, &rows, &cols}) if (!r->ok) return R::failure(r->error.status, r->error.message);
        unsigned __int128 total = (unsigned __int128)count.value * rows.value * cols.value;
        if (total * 8 != h.payload_len) return R::failure(INVALID, "designs.matrix payload length does not match count*rows*cols");
        auto m = std::make_shared<U64Matrices>();
        m->count = count.value; m->rows = rows.value; m->cols = cols.value;
        Reader r{h.payload, h.payload + h.payload_len};
        m->entries.reserve((size_t)total);
        for (uint64_t i = 0; i < (uint64_t)total; ++i) m->entries.push_back(r.u64());
        if (r.bad || r.p != r.end) return R::failure(INVALID, "designs.matrix payload length mismatch");
        o->u64_matrices = m;
        return R::success(o);
    }
    if (h.kind == "polytopes_small.vectors") {
        auto count = need(h, "count"), length = need(h, "length");
        for (auto *r : {&count, &length}) if (!r->ok) return R::failure(r->error.status, r->error.message);
        unsigned __int128 total = (unsigned __int128)count.value * length.value;
        if (total * 8 != h.payload_len) return R::failure(INVALID, "polytopes_small.vectors payload length does not match count*length");
        auto v = std::make_shared<U64Vectors>();
        v->count = count.value; v->length = length.value;
        Reader r{h.payload, h.payload + h.payload_len};
        v->entries.reserve((size_t)total);
        for (uint64_t i = 0; i < (uint64_t)total; ++i) v->entries.push_back(r.u64());
        if (r.bad || r.p != r.end) return R::failure(INVALID, "polytopes_small.vectors payload length mismatch");
        o->u64_vectors = v;
        return R::success(o);
    }
    if (h.kind == "lk.signed_matrices") {
        auto count = need(h, "count"), rows = need(h, "rows"), cols = need(h, "cols");
        for (auto *r : {&count, &rows, &cols}) if (!r->ok) return R::failure(r->error.status, r->error.message);
        unsigned __int128 total = (unsigned __int128)count.value * rows.value * cols.value;
        if (total * 8 != h.payload_len) return R::failure(INVALID, "lk.signed_matrices payload length does not match count*rows*cols");
        auto m = std::make_shared<I64Matrices>();
        m->count = count.value; m->rows = rows.value; m->cols = cols.value;
        Reader r{h.payload, h.payload + h.payload_len};
        m->entries.reserve((size_t)total);
        for (uint64_t i = 0; i < (uint64_t)total; ++i) m->entries.push_back(r.i64());
        if (r.bad || r.p != r.end) return R::failure(INVALID, "lk.signed_matrices payload length mismatch");
        o->i64_matrices = m;
        return R::success(o);
    }
    if (h.kind == "perm_groups.partition") {
        auto count = need(h, "count"), n = need(h, "n");
        for (auto *q : {&count, &n}) if (!q->ok) return R::failure(q->error.status, q->error.message);
        auto part = std::make_shared<Partitions>();
        part->count = count.value; part->n = n.value;
        Reader r{h.payload, h.payload + h.payload_len};
        r.entries(part->labels, count.value * n.value, 4);
        if (r.bad || r.p != r.end) return R::failure(INVALID, "perm_groups.partition payload length mismatch");
        for (Entry label : part->labels)
            if (label >= n.value) return R::failure(INVALID, "perm_groups.partition label is outside 0..n-1");
        o->partitions = part;
        return R::success(o);
    }
    if (h.kind == "perm_groups.bsgs") {
        auto count = need(h, "count"), n = need(h, "n");
        for (auto *q : {&count, &n}) if (!q->ok) return R::failure(q->error.status, q->error.message);
        if (n.value == 0) return R::failure(INVALID, "perm_groups.bsgs needs n >= 1");
        auto b = std::make_shared<Bsgs>();
        b->count = count.value; b->n = n.value;
        Reader r{h.payload, h.payload + h.payload_len};
        for (uint64_t i = 0; i <= count.value; ++i) b->base_offsets.push_back(r.u64());
        for (uint64_t i = 0; i <= count.value; ++i) b->strong_offsets.push_back(r.u64());
        auto offsets_valid = [count = count.value](const std::vector<uint64_t> &xs) {
            return xs.size() == count + 1 && xs.front() == 0 && std::is_sorted(xs.begin(), xs.end());
        };
        if (r.bad || !offsets_valid(b->base_offsets) || !offsets_valid(b->strong_offsets))
            return R::failure(INVALID, "perm_groups.bsgs offsets are malformed");
        r.entries(b->bases, b->base_offsets.back(), 4);
        r.entries(b->strong, b->strong_offsets.back() * n.value, 4);
        if (r.bad || r.p != r.end) return R::failure(INVALID, "perm_groups.bsgs payload length mismatch");
        for (Entry point : b->bases)
            if (point >= n.value) return R::failure(INVALID, "perm_groups.bsgs base point is outside 0..n-1");
        for (uint64_t i = 0; i < b->strong_offsets.back(); ++i) {
            std::vector<bool> seen(n.value, false);
            for (uint64_t j = 0; j < n.value; ++j) {
                Entry point = b->strong[i * n.value + j];
                if (point >= n.value || seen[point])
                    return R::failure(INVALID, "perm_groups.bsgs strong generator is not a permutation");
                seen[point] = true;
            }
        }
        o->bsgs = b;
        return R::success(o);
    }
    if (h.kind == "subgroups.lists") {
        auto count = need(h, "count");
        if (!count.ok) return R::failure(count.error.status, count.error.message);
        if (count.value == UINT64_MAX || (unsigned __int128)(count.value + 1) * 8 > h.payload_len)
            return R::failure(INVALID, "subgroups.lists group offsets are truncated");
        auto lists = std::make_shared<SubgroupLists>();
        lists->count = count.value;
        Reader r{h.payload, h.payload + h.payload_len};
        for (uint64_t i = 0; i <= count.value; ++i) lists->group_offsets.push_back(r.u64());
        if (r.bad || lists->group_offsets.front() != 0 ||
            !std::is_sorted(lists->group_offsets.begin(), lists->group_offsets.end()))
            return R::failure(INVALID, "subgroups.lists group offsets are malformed");
        uint64_t subgroups = lists->group_offsets.back();
        if (subgroups == UINT64_MAX || (unsigned __int128)(subgroups + 1) * 8 > (uint64_t)(r.end - r.p))
            return R::failure(INVALID, "subgroups.lists subgroup offsets are truncated");
        for (uint64_t i = 0; i <= subgroups; ++i) lists->subgroup_offsets.push_back(r.u64());
        if (r.bad || lists->subgroup_offsets.front() != 0 ||
            !std::is_sorted(lists->subgroup_offsets.begin(), lists->subgroup_offsets.end()) ||
            (unsigned __int128)lists->subgroup_offsets.back() * 8 != (uint64_t)(r.end - r.p))
            return R::failure(INVALID, "subgroups.lists subgroup offsets are malformed");
        for (uint64_t i = 0; i < lists->subgroup_offsets.back(); ++i) lists->elements.push_back(r.u64());
        if (r.bad || r.p != r.end)
            return R::failure(INVALID, "subgroups.lists payload length mismatch");
        for (uint64_t i = 0; i < subgroups; ++i) {
            auto first = lists->elements.begin() + lists->subgroup_offsets[i];
            auto last = lists->elements.begin() + lists->subgroup_offsets[i + 1];
            if (first == last || !std::is_sorted(first, last) || std::adjacent_find(first, last) != last)
                return R::failure(INVALID, "subgroups.lists subgroups must be nonempty increasing index lists");
        }
        o->subgroup_lists = std::move(lists);
        return R::success(o);
    }
    if (h.kind == "automorphisms.generators") {
        auto count = need(h, "count"), order = need(h, "order");
        for (auto *r : {&count, &order}) if (!r->ok) return R::failure(r->error.status, r->error.message);
        auto g = std::make_shared<PermutationGenerators>();
        g->count = count.value; g->order = order.value;
        Reader r{h.payload, h.payload + h.payload_len};
        for (uint64_t i = 0; i <= count.value; ++i) g->offsets.push_back(r.u64());
        if (r.bad || g->offsets.empty() || g->offsets[0] != 0)
            return R::failure(INVALID, "automorphisms.generators offsets are truncated or do not start at zero");
        for (uint64_t i = 1; i < g->offsets.size(); ++i)
            if (g->offsets[i] < g->offsets[i - 1])
                return R::failure(INVALID, "automorphisms.generators offsets are not increasing");
        r.entries(g->entries, g->offsets.back() * order.value, 4);
        if (r.bad || r.p != r.end) return R::failure(INVALID, "automorphisms.generators payload length mismatch");
        for (uint64_t i = 0; i < g->offsets.back(); ++i) {
            std::vector<uint8_t> seen(order.value, 0);
            for (uint64_t j = 0; j < order.value; ++j) {
                Entry e = g->entries[i * order.value + j];
                if (e >= order.value || seen[e]++)
                    return R::failure(INVALID, "automorphisms.generators payload contains a non-permutation");
            }
        }
        o->permutation_generators = g;
        return R::success(o);
    }
    if (h.kind == "coset_enumeration.representations") {
        auto count = need(h, "count"), generators = need(h, "generators"), max_cosets = need(h, "max_cosets");
        for (auto *r : {&count, &generators, &max_cosets})
            if (!r->ok) return R::failure(r->error.status, r->error.message);
        if (generators.value == 0 || generators.value >= (1ULL << 31) ||
            max_cosets.value == 0 || max_cosets.value >= (1ULL << 32))
            return R::failure(INVALID, "coset representation dimensions are out of range");
        unsigned __int128 image_count128 =
            (unsigned __int128)count.value * generators.value * max_cosets.value;
        unsigned __int128 words = count.value + image_count128;
        if (words * 4 != h.payload_len)
            return R::failure(INVALID, "coset_enumeration.representations payload length mismatch");
        uint64_t image_count = (uint64_t)image_count128;
        auto reps = std::make_shared<CosetRepresentations>();
        reps->count = count.value;
        reps->generators = generators.value;
        reps->max_cosets = max_cosets.value;
        Reader r{h.payload, h.payload + h.payload_len};
        r.entries(reps->degrees, count.value, 4);
        r.entries(reps->images, image_count, 4);
        if (r.bad || r.p != r.end)
            return R::failure(INVALID, "coset_enumeration.representations payload truncated");
        for (uint64_t i = 0; i < count.value; ++i) {
            Entry degree = reps->degrees[i];
            if (degree > max_cosets.value)
                return R::failure(INVALID, "coset representation degree exceeds max_cosets");
            for (uint64_t g = 0; g < generators.value; ++g) {
                uint64_t base = (i * generators.value + g) * max_cosets.value;
                std::vector<bool> seen(degree, false);
                for (uint64_t point = 0; point < degree; ++point) {
                    Entry image = reps->images[base + point];
                    if (image >= degree || seen[image])
                        return R::failure(INVALID, "coset representation generator is not a permutation");
                    seen[image] = true;
                }
                for (uint64_t point = degree; point < max_cosets.value; ++point)
                    if (reps->images[base + point] != 0)
                        return R::failure(INVALID, "coset representation padding must be zero");
            }
        }
        o->coset_representations = reps;
        return R::success(o);
    }
    if (h.kind == "posets.mobius") {
        auto count = need(h, "count");
        if (!count.ok) return R::failure(count.error.status, count.error.message);
        if (count.value == UINT64_MAX || count.value + 1 > h.payload_len / 8)
            return R::failure(INVALID, "posets.mobius offsets truncated");
        auto z = std::make_shared<SignedMatrices>();
        z->count = count.value;
        Reader r{h.payload, h.payload + h.payload_len};
        for (uint64_t i = 0; i <= count.value; ++i) z->offsets.push_back(r.u64());
        if (r.bad || z->offsets.front() != 0) return R::failure(INVALID, "posets.mobius invalid offsets");
        for (uint64_t i = 0; i < count.value; ++i) {
            if (z->offsets[i] > z->offsets[i + 1]) return R::failure(INVALID, "posets.mobius offsets are not monotone");
            uint64_t len = z->offsets[i + 1] - z->offsets[i];
            uint64_t n = static_cast<uint64_t>(std::sqrt(static_cast<long double>(len)));
            if (n * n != len) return R::failure(INVALID, "posets.mobius member is not square");
        }
        if (z->offsets.back() > (uint64_t)(r.end - r.p) / 8 || z->offsets.back() * 8 != (uint64_t)(r.end - r.p))
            return R::failure(INVALID, "posets.mobius payload length mismatch");
        for (uint64_t i = 0; i < z->offsets.back(); ++i) z->entries.push_back(std::bit_cast<int64_t>(r.u64()));
        o->signed_matrices = z;
        return R::success(o);
    }
    if (h.kind == "strongly_regular.params") {
        auto count = need(h, "count");
        if (!count.ok) return R::failure(count.error.status, count.error.message);
        if ((unsigned __int128)count.value * 33 != h.payload_len)
            return R::failure(INVALID, "strongly_regular.params payload length mismatch");
        auto ps = std::make_shared<SrgParams>();
        ps->count = count.value;
        ps->present.assign(h.payload, h.payload + count.value);
        Reader r{h.payload + count.value, h.payload + h.payload_len};
        for (uint64_t i = 0; i < count.value * 4; ++i) ps->values.push_back(r.u64());
        if (r.bad || r.p != r.end || std::any_of(ps->present.begin(), ps->present.end(), [](uint8_t x) { return x > 1; }))
            return R::failure(INVALID, "invalid strongly_regular.params payload");
        o->srg_params = ps;
        return R::success(o);
    }
    if (h.kind == "strongly_regular.spectra") {
        auto count = need(h, "count");
        if (!count.ok) return R::failure(count.error.status, count.error.message);
        if ((unsigned __int128)count.value * 49 != h.payload_len)
            return R::failure(INVALID, "strongly_regular.spectra payload length mismatch");
        auto ss = std::make_shared<SrgSpectra>();
        ss->count = count.value;
        ss->present.assign(h.payload, h.payload + count.value);
        Reader r{h.payload + count.value, h.payload + h.payload_len};
        for (uint64_t i = 0; i < count.value; ++i) {
            ss->k.push_back(r.u64());
            ss->delta_negative.push_back(r.u64());
            ss->delta_abs.push_back(r.u64());
            ss->discriminant.push_back(r.u64());
            ss->multiplicity_plus.push_back(r.u64());
            ss->multiplicity_minus.push_back(r.u64());
        }
        if (r.bad || r.p != r.end ||
            std::any_of(ss->present.begin(), ss->present.end(), [](uint8_t x) { return x > 1; }) ||
            std::any_of(ss->delta_negative.begin(), ss->delta_negative.end(), [](uint64_t x) { return x > 1; }))
            return R::failure(INVALID, "invalid strongly_regular.spectra payload");
        o->srg_spectra = ss;
        return R::success(o);
    }
    return R::failure(INVALID, "unknown object kind " + h.kind);
}

} // namespace

unsigned entry_width(uint64_t p) {
    if (p == 0 || p == NATURALS || p == GRAMS) return 4;
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
    if (matrix && matrix->p == 0) return {{"n", matrix->cols}, {"count", matrix->count * matrix->rows}};
    if (matrix && matrix->p == NATURALS) return {{"count", matrix->count}, {"rows", matrix->rows}, {"cols", matrix->cols}};
    if (matrix && matrix->p == GRAMS) return {{"count", matrix->count}, {"n", matrix->rows}};
    if (matrix) return {{"p", matrix->p}, {"count", matrix->count}, {"rows", matrix->rows}, {"cols", matrix->cols}};
    if (basis) return {{"p", basis->p}, {"count", basis->count}, {"cols", basis->cols}};
    if (graph_groups) return {{"count", graph_groups->count}, {"n", graph_groups->n}};
    if (regular_subgroups) return {{"count", regular_subgroups->count}, {"n", regular_subgroups->n}};
    if (solutions) return {{"p", solutions->p}, {"count", solutions->count}, {"length", solutions->length}};
    if (inverses) return {{"p", inverses->p}, {"count", inverses->count}, {"n", inverses->n}};
    if (witness) return {{"p", witness->p}, {"count", witness->count}, {"rows", witness->rows}, {"cols", witness->cols}};
    if (elements) return {{"p", elements->p}, {"count", elements->count}};
    if (degrees) return {{"count", degrees->count}};
    if (continued_fractions) return {{"count", continued_fractions->count}};
    if (quadratic_units) return {{"count", quadratic_units->count}};
    if (factorisation) return {{"count", factorisation->count}};
    if (cycle_index) return {{"degree", cycle_index->degree}, {"count", cycle_index->multiplicities.size()},
                             {"denominator", cycle_index->denominator}};
    if (spectra) return {{"n", spectra->n}, {"count", spectra->count}};
    if (u64_matrices) return {{"count", u64_matrices->count}, {"rows", u64_matrices->rows}, {"cols", u64_matrices->cols}};
    if (u64_vectors) return {{"count", u64_vectors->count}, {"length", u64_vectors->length}};
    if (i64_matrices) return {{"count", i64_matrices->count}, {"rows", i64_matrices->rows}, {"cols", i64_matrices->cols}};
    if (partitions) return {{"count", partitions->count}, {"n", partitions->n}};
    if (bsgs) return {{"count", bsgs->count}, {"n", bsgs->n}};
    if (character_table) return {{"order", character_table->order}, {"classes", character_table->classes},
                                 {"conductor", character_table->conductor}};
    if (character_indicators) return {{"count", character_indicators->values.size()}};
    if (permutation_generators) return {{"count", permutation_generators->count}, {"order", permutation_generators->order}};
    if (subgroup_lists) return {{"count", subgroup_lists->count}};
    if (weight_enumerators) return {{"count", weight_enumerators->count}, {"n", weight_enumerators->n}};
    if (signed_matrices) return {{"count", signed_matrices->count}};
    if (characters) return {{"count", characters->values.size()}};
    if (rsk_pairs) return {{"count", rsk_pairs->count}, {"length", rsk_pairs->length}};
    if (curve_groups) return {{"count", curve_groups->count}};
    if (coefficients) return {{"count", coefficients->count}, {"length", coefficients->length}};
    if (srg_params) return {{"count", srg_params->count}};
    if (srg_spectra) return {{"count", srg_spectra->count}};
    if (coset_representations) return {{"count", coset_representations->count},
                                       {"generators", coset_representations->generators},
                                       {"max_cosets", coset_representations->max_cosets}};
    if (integers) return {{"count", integers->values.size()}};
    if (degree_sequences) return {{"count", degree_sequences->count}, {"n", degree_sequences->n}};
    if (theta_series) return {{"count", theta_series->count}, {"bound", theta_series->bound}};
    if (short_vectors) return {{"count", short_vectors->count}, {"n", short_vectors->n}, {"bound", short_vectors->bound}};
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
    } else if (o.graph_groups) {
        w.u64s(o.graph_groups->offsets);
        w.entries(o.graph_groups->entries, 4);
        write_header(out, "graph_iso.groups", o.params(), w.out);
    } else if (o.regular_subgroups) {
        w.u64s(o.regular_subgroups->offsets);
        w.entries(o.regular_subgroups->entries, 4);
        write_header(out, "vertex_transitive.regular_subgroups", o.params(), w.out);
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
    } else if (o.elements) {
        w.u64s(o.elements->offsets);
        w.entries(o.elements->values, entry_width(o.elements->p));
        write_header(out, "polynomials_fq.elements", o.params(), w.out);
    } else if (o.degrees) {
        w.u64s(o.degrees->offsets);
        w.u64s(o.degrees->values);
        write_header(out, "polynomials_fq.degrees", o.params(), w.out);
    } else if (o.continued_fractions) {
        w.u64s(o.continued_fractions->offsets);
        w.u64s(o.continued_fractions->values);
        write_header(out, "continued_fractions_and_pell.expansion", o.params(), w.out);
    } else if (o.quadratic_units) {
        w.bytes(o.quadratic_units->solvable);
        w.bytes(o.quadratic_units->negative);
        w.u64s(o.quadratic_units->pairs);
        write_header(out, "continued_fractions_and_pell.unit", o.params(), w.out);
    } else if (o.degree_sequences) {
        w.entries(o.degree_sequences->entries, 4);
        write_header(out, "graphs.degree_sequences", o.params(), w.out);
    } else if (o.factorisation) {
        w.u64s(o.factorisation->offsets);
        w.u64s(o.factorisation->pairs);
        write_header(out, "sieve_ranges.factorisation", o.params(), w.out);
    } else if (o.cycle_index) {
        for (uint64_t i = 0; i < o.cycle_index->multiplicities.size(); ++i) {
            w.u64(o.cycle_index->multiplicities[i]);
            for (uint64_t j = 0; j < o.cycle_index->degree; ++j)
                w.u64(o.cycle_index->cycles[i * o.cycle_index->degree + j]);
        }
        write_header(out, "burnside.cycle_index", o.params(), w.out);
    } else if (o.spectra) {
        w.u64s(o.spectra->offsets);
        w.entries(o.spectra->exponents, 4);
        write_header(out, "circulants.spectra", o.params(), w.out);
    } else if (o.u64_matrices) {
        w.u64s(o.u64_matrices->entries);
        write_header(out, "designs.matrix", o.params(), w.out);
    } else if (o.u64_vectors) {
        w.u64s(o.u64_vectors->entries);
        write_header(out, "polytopes_small.vectors", o.params(), w.out);
    } else if (o.i64_matrices) {
        w.i64s(o.i64_matrices->entries);
        write_header(out, "lk.signed_matrices", o.params(), w.out);
    } else if (o.partitions) {
        w.entries(o.partitions->labels, 4);
        write_header(out, "perm_groups.partition", o.params(), w.out);
    } else if (o.bsgs) {
        w.u64s(o.bsgs->base_offsets);
        w.u64s(o.bsgs->strong_offsets);
        w.entries(o.bsgs->bases, 4);
        w.entries(o.bsgs->strong, 4);
        write_header(out, "perm_groups.bsgs", o.params(), w.out);
    } else if (o.character_table) {
        w.u64s(o.character_table->representatives);
        w.u64s(o.character_table->class_sizes);
        w.u64s(o.character_table->degrees);
        w.entries(o.character_table->spectra, 4);
        write_header(out, "characters.table", o.params(), w.out);
    } else if (o.character_indicators) {
        for (int8_t value : o.character_indicators->values) w.out.push_back(static_cast<uint8_t>(value));
        write_header(out, "characters.indicators", o.params(), w.out);
    } else if (o.permutation_generators) {
        w.u64s(o.permutation_generators->offsets);
        w.entries(o.permutation_generators->entries, 4);
        write_header(out, "automorphisms.generators", o.params(), w.out);
    } else if (o.subgroup_lists) {
        w.u64s(o.subgroup_lists->group_offsets);
        w.u64s(o.subgroup_lists->subgroup_offsets);
        w.u64s(o.subgroup_lists->elements);
        write_header(out, "subgroups.lists", o.params(), w.out);
    } else if (o.weight_enumerators) {
        w.u64s(o.weight_enumerators->coefficients);
        write_header(out, "linear_codes.weight_enumerators", o.params(), w.out);
    } else if (o.signed_matrices) {
        w.u64s(o.signed_matrices->offsets);
        w.i64s(o.signed_matrices->entries);
        write_header(out, "posets.mobius", o.params(), w.out);
    } else if (o.characters) {
        w.i64s(o.characters->values);
        write_header(out, "young.characters", o.params(), w.out);
    } else if (o.rsk_pairs) {
        w.entries(o.rsk_pairs->shapes, 4);
        w.entries(o.rsk_pairs->insertion, 4);
        w.entries(o.rsk_pairs->recording, 4);
        write_header(out, "young.rsk_pairs", o.params(), w.out);
    } else if (o.curve_groups) {
        w.u64s(o.curve_groups->orders);
        write_header(out, "elliptic_curves_fp.group", o.params(), w.out);
    } else if (o.coefficients) {
        w.i64s(o.coefficients->values);
        write_header(out, "graph_polynomials.coefficients", o.params(), w.out);
    } else if (o.srg_params) {
        w.bytes(o.srg_params->present);
        w.u64s(o.srg_params->values);
        write_header(out, "strongly_regular.params", o.params(), w.out);
    } else if (o.srg_spectra) {
        w.bytes(o.srg_spectra->present);
        for (uint64_t i = 0; i < o.srg_spectra->count; ++i) {
            w.u64(o.srg_spectra->k[i]);
            w.u64(o.srg_spectra->delta_negative[i]);
            w.u64(o.srg_spectra->delta_abs[i]);
            w.u64(o.srg_spectra->discriminant[i]);
            w.u64(o.srg_spectra->multiplicity_plus[i]);
            w.u64(o.srg_spectra->multiplicity_minus[i]);
        }
        write_header(out, "strongly_regular.spectra", o.params(), w.out);
    } else if (o.coset_representations) {
        w.entries(o.coset_representations->degrees, 4);
        w.entries(o.coset_representations->images, 4);
        write_header(out, "coset_enumeration.representations", o.params(), w.out);
    } else if (o.integers) {
        w.u64s(o.integers->values);
        const char *kind = o.kind == "burnside.counts" ? "burnside.counts" :
                           o.kind == "characters.multiplicities" ? "characters.multiplicities" : "integers";
        write_header(out, kind, o.params(), w.out);
    } else if (o.theta_series) {
        w.u64s(o.theta_series->coefficients);
        write_header(out, "lattices.theta_series", o.params(), w.out);
    } else if (o.short_vectors) {
        w.u64s(o.short_vectors->offsets);
        w.entries(o.short_vectors->entries, 4);
        write_header(out, "lattices.short_vectors", o.params(), w.out);
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
