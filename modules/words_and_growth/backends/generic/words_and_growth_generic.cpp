/* Exact growth for presentations with finite complete shortlex rewriting systems. */
#include "../../../../runtime/src/reduce.hpp"

#include <limits>

namespace lk::words_and_growth {
namespace {

using R = Result<std::shared_ptr<Object>>;
using Word = std::vector<Entry>;
constexpr int INVALID = 1;
constexpr uint64_t MAX_NORMALISE_STEPS = 1ULL << 20;
constexpr uint64_t MAX_CRITICAL_PAIRS = 1ULL << 20;
constexpr uint64_t MAX_ENUMERATED_WORDS = 1ULL << 24;

struct Rule {
    Word lhs;
    Word rhs;
    bool operator==(const Rule &other) const { return lhs == other.lhs && rhs == other.rhs; }
};

bool shortlex_less(const Word &a, const Word &b) {
    if (a.size() != b.size()) return a.size() < b.size();
    return std::lexicographical_compare(a.begin(), a.end(), b.begin(), b.end());
}

Word replace_at(const Word &word, size_t pos, size_t length, const Word &replacement) {
    Word out;
    out.reserve(word.size() - length + replacement.size());
    out.insert(out.end(), word.begin(), word.begin() + pos);
    out.insert(out.end(), replacement.begin(), replacement.end());
    out.insert(out.end(), word.begin() + pos + length, word.end());
    return out;
}

bool rewrite_once(const std::vector<Rule> &rules, const Word &word, Word &out) {
    for (size_t pos = 0; pos <= word.size(); ++pos) {
        for (const Rule &rule : rules) {
            if (pos + rule.lhs.size() > word.size()) continue;
            if (std::equal(rule.lhs.begin(), rule.lhs.end(), word.begin() + pos)) {
                out = replace_at(word, pos, rule.lhs.size(), rule.rhs);
                return true;
            }
        }
    }
    return false;
}

Result<Word> normalise(const std::vector<Rule> &rules, Word word) {
    Word next;
    for (uint64_t step = 0; step < MAX_NORMALISE_STEPS; ++step) {
        if (!rewrite_once(rules, word, next)) return Result<Word>::success(std::move(word));
        word.swap(next);
    }
    return Result<Word>::failure(INVALID, "normal-form computation exceeded the step limit");
}

Status check_joinable(const std::vector<Rule> &rules, const Word &left, const Word &right) {
    auto a = normalise(rules, left);
    if (!a.ok) return fail(a.error.status, a.error.message);
    auto b = normalise(rules, right);
    if (!b.ok) return fail(b.error.status, b.error.message);
    if (a.value != b.value) return fail(INVALID, "rewriting system is not confluent: a critical pair does not join");
    return ok();
}

Status check_critical_pairs(const std::vector<Rule> &rules) {
    uint64_t checked = 0;
    auto account = [&]() -> Status {
        if (++checked > MAX_CRITICAL_PAIRS)
            return fail(INVALID, "rewriting system has too many critical pairs");
        return ok();
    };
    for (const Rule &outer : rules) {
        for (const Rule &inner : rules) {
            if (inner.lhs.size() <= outer.lhs.size()) {
                for (size_t pos = 0; pos + inner.lhs.size() <= outer.lhs.size(); ++pos) {
                    if (!std::equal(inner.lhs.begin(), inner.lhs.end(), outer.lhs.begin() + pos)) continue;
                    auto count = account();
                    if (!count.ok) return count;
                    auto joined = check_joinable(rules, outer.rhs,
                                                 replace_at(outer.lhs, pos, inner.lhs.size(), inner.rhs));
                    if (!joined.ok) return joined;
                }
            }
            size_t overlap_max = std::min(outer.lhs.size(), inner.lhs.size());
            for (size_t overlap = 1; overlap <= overlap_max; ++overlap) {
                if (!std::equal(outer.lhs.end() - overlap, outer.lhs.end(), inner.lhs.begin())) continue;
                auto count = account();
                if (!count.ok) return count;
                Word left = outer.rhs;
                left.insert(left.end(), inner.lhs.begin() + overlap, inner.lhs.end());
                Word right(outer.lhs.begin(), outer.lhs.end() - overlap);
                right.insert(right.end(), inner.rhs.begin(), inner.rhs.end());
                auto joined = check_joinable(rules, left, right);
                if (!joined.ok) return joined;
            }
        }
    }
    return ok();
}

struct Presentation {
    uint64_t generators;
    uint64_t alphabet;
    std::vector<Rule> rules;
    bool strictly_length_reducing;
};

Result<Presentation> presentation(const Request &req) {
    auto git = req.int_args.find("generators");
    if (git == req.int_args.end()) return Result<Presentation>::failure(INVALID, "missing generators");
    uint64_t generators = git->second;
    if (generators == 0 || generators > 128)
        return Result<Presentation>::failure(INVALID, "generators must be between 1 and 128");
    uint64_t alphabet = 2 * generators;

    auto rit = req.handle_args.find("relations");
    if (rit == req.handle_args.end() || !rit->second->matrix)
        return Result<Presentation>::failure(INVALID, "relations must be an lk.naturals matrix of encoded equations");
    const Matrix &encoded = *rit->second->matrix;
    if (encoded.p != NATURALS)
        return Result<Presentation>::failure(INVALID, "relations must be an lk.naturals matrix of encoded equations");
    uint64_t count = 0, width = encoded.cols;
    bool batched_rows = encoded.rows == 1;
    if (batched_rows) count = encoded.count;
    else if (encoded.count == 1) count = encoded.rows;
    else return Result<Presentation>::failure(INVALID, "relations must be vectors of one common width");
    if (count > 256 || width > 514)
        return Result<Presentation>::failure(INVALID, "presentation exceeds the equation count or width limit");

    std::vector<Rule> rules;
    auto add_rule = [&](Word left, Word right) {
        if (left == right) return;
        Rule rule = shortlex_less(left, right) ? Rule{std::move(right), std::move(left)}
                                               : Rule{std::move(left), std::move(right)};
        if (std::find(rules.begin(), rules.end(), rule) == rules.end()) rules.push_back(std::move(rule));
    };
    for (uint64_t i = 0; i < count; ++i) {
        const Entry *row = batched_rows ? encoded.at(i) : encoded.entries.data() + i * width;
        if (width < 2) return Result<Presentation>::failure(INVALID, "relation row needs two lengths");
        uint64_t left_length = row[0], right_length = row[1];
        if (left_length + right_length > width - 2)
            return Result<Presentation>::failure(INVALID, "relation lengths exceed the row width");
        Word left(row + 2, row + 2 + left_length);
        Word right(row + 2 + left_length, row + 2 + left_length + right_length);
        for (Entry letter : left)
            if (letter >= alphabet) return Result<Presentation>::failure(INVALID, "relation has a symbol outside the generator alphabet");
        for (Entry letter : right)
            if (letter >= alphabet) return Result<Presentation>::failure(INVALID, "relation has a symbol outside the generator alphabet");
        for (uint64_t j = 2 + left_length + right_length; j < width; ++j)
            if (row[j] != alphabet) return Result<Presentation>::failure(INVALID, "relation padding must equal the alphabet size");
        add_rule(std::move(left), std::move(right));
    }
    for (uint64_t i = 0; i < generators; ++i) {
        add_rule({(Entry)(2 * i), (Entry)(2 * i + 1)}, {});
        add_rule({(Entry)(2 * i + 1), (Entry)(2 * i)}, {});
    }
    auto complete = check_critical_pairs(rules);
    if (!complete.ok) return Result<Presentation>::failure(complete.error.status, complete.error.message);
    bool strict = std::all_of(rules.begin(), rules.end(), [](const Rule &rule) { return rule.rhs.size() < rule.lhs.size(); });
    return Result<Presentation>::success(Presentation{generators, alphabet, std::move(rules), strict});
}

bool ends_with(const Word &word, const Word &suffix) {
    return suffix.size() <= word.size() && std::equal(suffix.begin(), suffix.end(), word.end() - suffix.size());
}

struct Growth {
    std::vector<uint64_t> sphere;
    std::vector<uint64_t> ball;
};

Result<Growth> growth_counts(const Presentation &p, uint64_t max_radius) {
    if (max_radius > 1000000) return Result<Growth>::failure(INVALID, "radius exceeds the generic backend limit of 1000000");
    std::vector<Word> prefixes(1);
    for (const Rule &rule : p.rules) {
        for (size_t length = 1; length < rule.lhs.size(); ++length) {
            Word prefix(rule.lhs.begin(), rule.lhs.begin() + length);
            if (std::find(prefixes.begin(), prefixes.end(), prefix) == prefixes.end()) prefixes.push_back(std::move(prefix));
        }
    }
    std::vector<std::vector<int32_t>> transition(prefixes.size(), std::vector<int32_t>(p.alphabet, -1));
    for (size_t state = 0; state < prefixes.size(); ++state) {
        for (uint64_t letter = 0; letter < p.alphabet; ++letter) {
            Word candidate = prefixes[state];
            candidate.push_back((Entry)letter);
            bool reducible = std::any_of(p.rules.begin(), p.rules.end(), [&](const Rule &rule) { return ends_with(candidate, rule.lhs); });
            if (reducible) continue;
            size_t best = 0;
            for (size_t next = 1; next < prefixes.size(); ++next)
                if (prefixes[next].size() > prefixes[best].size() && ends_with(candidate, prefixes[next])) best = next;
            transition[state][letter] = (int32_t)best;
        }
    }

    Growth out;
    out.sphere.assign(max_radius + 1, 0);
    out.ball.assign(max_radius + 1, 0);
    out.sphere[0] = out.ball[0] = 1;
    std::vector<uint64_t> current(prefixes.size(), 0), next(prefixes.size(), 0);
    current[0] = 1;
    for (uint64_t radius = 1; radius <= max_radius; ++radius) {
        std::fill(next.begin(), next.end(), 0);
        for (size_t state = 0; state < prefixes.size(); ++state) {
            if (!current[state]) continue;
            for (uint64_t letter = 0; letter < p.alphabet; ++letter) {
                int32_t target = transition[state][letter];
                if (target < 0) continue;
                unsigned __int128 value = (unsigned __int128)next[target] + current[state];
                if (value > UINT64_MAX) return Result<Growth>::failure(INVALID, "growth coefficient does not fit in 64 bits");
                next[target] = (uint64_t)value;
            }
        }
        unsigned __int128 sphere = 0;
        for (uint64_t value : next) sphere += value;
        if (sphere > UINT64_MAX) return Result<Growth>::failure(INVALID, "growth coefficient does not fit in 64 bits");
        out.sphere[radius] = (uint64_t)sphere;
        unsigned __int128 ball = (unsigned __int128)out.ball[radius - 1] + out.sphere[radius];
        if (ball > UINT64_MAX) return Result<Growth>::failure(INVALID, "ball size does not fit in 64 bits");
        out.ball[radius] = (uint64_t)ball;
        current.swap(next);
    }
    return Result<Growth>::success(std::move(out));
}

Result<uint64_t> checked_word_count(uint64_t alphabet, uint64_t length) {
    uint64_t value = 1;
    for (uint64_t i = 0; i < length; ++i) {
        if (value > UINT64_MAX / alphabet) return Result<uint64_t>::failure(INVALID, "word count does not fit in 64 bits");
        value *= alphabet;
    }
    return Result<uint64_t>::success(value);
}

Word unrank_word(uint64_t index, uint64_t alphabet, uint64_t length) {
    Word word(length, 0);
    for (uint64_t i = length; i > 0; --i) {
        word[i - 1] = (Entry)(index % alphabet);
        index /= alphabet;
    }
    return word;
}

Result<uint64_t> geodesic_count(const Presentation &p, uint64_t radius, uint64_t sphere) {
    if (p.strictly_length_reducing) return Result<uint64_t>::success(sphere);
    auto total = checked_word_count(p.alphabet, radius);
    if (!total.ok) return total;
    if (total.value > MAX_ENUMERATED_WORDS)
        return Result<uint64_t>::failure(INVALID, "geodesic_count needs to enumerate more than 2^24 words for an equal-length rewriting system");
    uint64_t count = 0;
    for (uint64_t index = 0; index < total.value; ++index) {
        auto nf = normalise(p.rules, unrank_word(index, p.alphabet, radius));
        if (!nf.ok) return Result<uint64_t>::failure(nf.error.status, nf.error.message);
        count += nf.value.size() == radius;
    }
    return Result<uint64_t>::success(count);
}

R run_range(const Request &req, const Presentation &p) {
    const Family &family = *req.family;
    uint64_t max_radius = family.b - 1;
    auto growth = growth_counts(p, max_radius);
    if (!growth.ok) return R::failure(growth.error.status, growth.error.message);
    auto size = family.size();
    if (!size.ok) return R::failure(size.error.status, size.error.message);
    Reduction reduction = parse_reduction(req.reduction);
    Shared shared;
    auto prepared = prepare_all(reduction, size.value, shared);
    if (!prepared.ok) return R::failure(prepared.error.status, prepared.error.message);
    Accumulator accumulator(reduction, &shared);
    for (uint64_t index = 0; index < size.value; ++index) {
        uint64_t radius = family.a + index;
        uint64_t value = 0;
        if (req.op == "ball_size") value = growth.value.ball[radius];
        else if (req.op == "sphere_size") value = growth.value.sphere[radius];
        else {
            auto count = geodesic_count(p, radius, growth.value.sphere[radius]);
            if (!count.ok) return R::failure(count.error.status, count.error.message);
            value = count.value;
        }
        accumulator.integer(index, value);
    }
    std::vector<Accumulator> accumulators;
    accumulators.push_back(std::move(accumulator));
    return assemble(req, reduction, accumulators, shared);
}

R run_words(const Request &req, const Presentation &p) {
    const Family &family = *req.family;
    if (family.p != p.alphabet)
        return R::failure(INVALID, "words family alphabet must equal 2 * generators");
    auto size = family.size();
    if (!size.ok) return R::failure(size.error.status, size.error.message);
    Reduction reduction = parse_reduction(req.reduction);
    Shared shared;
    auto prepared = prepare_all(reduction, size.value, shared);
    if (!prepared.ok) return R::failure(prepared.error.status, prepared.error.message);
    uint32_t threads = std::max<uint32_t>(1, std::min<uint64_t>(req.threads, size.value ? size.value : 1));
    std::vector<Accumulator> accumulators;
    for (uint32_t thread = 0; thread < threads; ++thread) accumulators.emplace_back(reduction, &shared);
    auto statuses = parallel_ranges(size.value, threads, [&](uint32_t thread, uint64_t begin, uint64_t end) -> Status {
        for (uint64_t index = begin; index < end; ++index) {
            if (accumulators[thread].exhausted(index)) break;
            Word word = unrank_word(index, p.alphabet, family.n);
            auto nf = normalise(p.rules, word);
            if (!nf.ok) return fail(nf.error.status, nf.error.message);
            accumulators[thread].boolean(index, nf.value.size() == word.size());
        }
        return ok();
    });
    for (const Status &status : statuses)
        if (!status.ok) return R::failure(status.error.status, status.error.message);
    return assemble(req, reduction, accumulators, shared);
}

R run(const Request &req) {
    auto p = presentation(req);
    if (!p.ok) return R::failure(p.error.status, p.error.message);
    if (req.op == "is_geodesic") return run_words(req, p.value);
    return run_range(req, p.value);
}

BackendRegistration registration{Backend{
    "words_and_growth", "generic",
    [] { return true; },
    [](const Request &) { return true; },
    run,
    0}};

} // namespace
} // namespace lk::words_and_growth
