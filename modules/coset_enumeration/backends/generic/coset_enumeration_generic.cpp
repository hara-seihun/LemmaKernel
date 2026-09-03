#include "../../../../runtime/src/reduce.hpp"

#include <deque>

namespace lk::coset_enumeration {
namespace {

using R = Result<std::shared_ptr<Object>>;
constexpr int INVALID = 1;
constexpr Entry NONE = UINT32_MAX;

struct Answer {
    Entry degree = 0;
    std::vector<Entry> images;
};

struct Table {
    enum class TaskKind { Link, Merge };
    struct Task { TaskKind kind; Entry a, letter, b; };

    Entry generators, letters, bound;
    std::vector<Entry> table;
    std::vector<Entry> parent;
    bool failed = false;

    Table(uint64_t generator_count, uint64_t max_cosets)
        : generators((Entry)generator_count), letters((Entry)(2 * generator_count)), bound((Entry)max_cosets),
          table(2 * generator_count, NONE), parent{0} {}

    Entry root(Entry coset) const {
        while (parent[coset] != coset) coset = parent[coset];
        return coset;
    }

    Entry cell(Entry coset, Entry letter) const {
        Entry value = table[(uint64_t)root(coset) * letters + letter];
        return value == NONE ? NONE : root(value);
    }

    Entry add_coset() {
        if (parent.size() >= bound) {
            failed = true;
            return NONE;
        }
        Entry coset = (Entry)parent.size();
        parent.push_back(coset);
        table.insert(table.end(), letters, NONE);
        return coset;
    }

    void process(std::deque<Task> tasks) {
        uint64_t steps = 0;
        unsigned __int128 budget128 = 1024 + (unsigned __int128)16 * (bound + 1ULL) * (bound + 1ULL) * (letters + 1ULL);
        uint64_t budget = budget128 > UINT64_MAX ? UINT64_MAX : (uint64_t)budget128;
        while (!tasks.empty() && !failed) {
            if (++steps > budget) {
                failed = true;
                return;
            }
            Task task = tasks.front();
            tasks.pop_front();
            if (task.kind == TaskKind::Merge) {
                Entry ra = root(task.a), rb = root(task.b);
                if (ra == rb) continue;
                Entry lo = std::min(ra, rb), hi = std::max(ra, rb);
                parent[hi] = lo;
                std::deque<Task> next;
                for (Entry letter = 0; letter < letters; ++letter) {
                    Entry value = table[(uint64_t)hi * letters + letter];
                    if (value != NONE) next.push_back({TaskKind::Link, lo, letter, value});
                }
                next.insert(next.end(), tasks.begin(), tasks.end());
                tasks.swap(next);
                continue;
            }

            Entry rc = root(task.a), rd = root(task.b), letter = task.letter;
            Entry forward = cell(rc, letter), backward = cell(rd, letter ^ 1);
            if (forward != NONE && forward != rd) {
                tasks.push_front({TaskKind::Link, rc, letter, rd});
                tasks.push_front({TaskKind::Merge, forward, 0, rd});
            } else if (backward != NONE && backward != rc) {
                tasks.push_front({TaskKind::Link, rc, letter, rd});
                tasks.push_front({TaskKind::Merge, backward, 0, rc});
            } else {
                table[(uint64_t)rc * letters + letter] = rd;
                table[(uint64_t)rd * letters + (letter ^ 1)] = rc;
            }
        }
    }

    void link(Entry a, Entry letter, Entry b) {
        process({Task{TaskKind::Link, a, letter, b}});
    }

    void merge(Entry a, Entry b) {
        process({Task{TaskKind::Merge, a, 0, b}});
    }

    void enforce(Entry start, const std::vector<Entry> &word, Entry end) {
        uint64_t fuel = (uint64_t)bound + word.size() + 2;
        while (fuel-- && !failed) {
            Entry c = root(start);
            size_t i = 0;
            while (i < word.size()) {
                Entry next = cell(c, word[i]);
                if (next == NONE) break;
                c = next;
                ++i;
            }

            Entry d = root(end);
            size_t j = word.size();
            while (j > i) {
                Entry previous = cell(d, word[j - 1] ^ 1);
                if (previous == NONE) break;
                d = previous;
                --j;
            }

            if (j == i) {
                merge(c, d);
                return;
            }
            if (j == i + 1) {
                link(c, word[i], d);
                return;
            }
            Entry fresh = add_coset();
            if (failed) return;
            link(c, word[i], fresh);
        }
        if (!failed) failed = true;
    }

    Answer enumerate(const std::vector<std::vector<Entry>> &relators,
                     const std::vector<std::vector<Entry>> &subgroup) {
        for (const auto &word : subgroup) enforce(0, word, 0);
        for (uint64_t cursor = 0; cursor < parent.size() && !failed; ++cursor) {
            if (root((Entry)cursor) != cursor) continue;
            for (const auto &word : relators) {
                enforce((Entry)cursor, word, (Entry)cursor);
                if (failed) break;
            }
            for (Entry generator = 0; generator < generators && !failed; ++generator) {
                Entry c = root((Entry)cursor);
                if (cell(c, 2 * generator) != NONE) continue;
                Entry fresh = add_coset();
                if (!failed) link(c, 2 * generator, fresh);
            }
        }
        if (failed) return {};

        std::vector<Entry> roots;
        for (Entry i = 0; i < parent.size(); ++i)
            if (root(i) == i) roots.push_back(i);
        std::vector<Entry> position(parent.size(), NONE);
        for (Entry i = 0; i < roots.size(); ++i) position[roots[i]] = i;
        Answer answer;
        answer.degree = (Entry)roots.size();
        answer.images.reserve((uint64_t)generators * answer.degree);
        for (Entry generator = 0; generator < generators; ++generator)
            for (Entry c : roots) answer.images.push_back(position[cell(c, 2 * generator)]);
        return answer;
    }
};

using WordsResult = Result<std::vector<std::vector<Entry>>>;

WordsResult natural_vectors_arg(const Request &req, const char *name) {
    auto it = req.handle_args.find(name);
    if (it == req.handle_args.end() || !it->second->matrix || it->second->matrix->p != NATURALS)
        return WordsResult::failure(INVALID, std::string("`") + name + "` must be natural-number vectors");
    const Matrix &m = *it->second->matrix;
    std::vector<std::vector<Entry>> rows;
    if (m.rows == 1) {
        rows.reserve(m.count);
        for (uint64_t i = 0; i < m.count; ++i) rows.emplace_back(m.at(i), m.at(i) + m.cols);
    } else if (m.count == 1) {
        rows.reserve(m.rows);
        for (uint64_t i = 0; i < m.rows; ++i)
            rows.emplace_back(m.entries.begin() + i * m.cols, m.entries.begin() + (i + 1) * m.cols);
    } else {
        return WordsResult::failure(INVALID,
            std::string("`") + name + "` must be a batch of 1 x width vectors or one rows x width matrix");
    }
    return WordsResult::success(std::move(rows));
}

WordsResult subgroup_words_arg(const Request &req, uint64_t letters) {
    auto raw = natural_vectors_arg(req, "subgroup");
    if (!raw.ok) return raw;
    std::vector<std::vector<Entry>> words;
    words.reserve(raw.value.size());
    for (const auto &row : raw.value) {
        auto padding = std::find_if(row.begin(), row.end(), [&](Entry value) { return value >= letters; });
        if (std::any_of(padding, row.end(), [&](Entry value) { return value != letters; }))
            return WordsResult::failure(INVALID,
                "subgroup word contains an invalid letter or non-trailing padding");
        words.emplace_back(row.begin(), padding);
    }
    return WordsResult::success(std::move(words));
}

WordsResult relation_relators_arg(const Request &req, uint64_t letters) {
    auto encoded = natural_vectors_arg(req, "relations");
    if (!encoded.ok) return encoded;
    std::vector<std::vector<Entry>> relators;
    relators.reserve(encoded.value.size());
    for (const auto &row : encoded.value) {
        if (row.size() < 2)
            return WordsResult::failure(INVALID, "relation row needs two lengths");
        uint64_t left_length = row[0], right_length = row[1];
        if (left_length > row.size() - 2 || right_length > row.size() - 2 - left_length)
            return WordsResult::failure(INVALID, "relation lengths exceed the row width");
        uint64_t used = 2 + left_length + right_length;
        for (uint64_t i = 2; i < used; ++i)
            if (row[i] >= letters)
                return WordsResult::failure(INVALID,
                    "relation has a symbol outside the generator alphabet");
        for (uint64_t i = used; i < row.size(); ++i)
            if (row[i] != letters)
                return WordsResult::failure(INVALID, "relation padding must equal the alphabet size");
        std::vector<Entry> relator(row.begin() + 2, row.begin() + 2 + left_length);
        relator.reserve(left_length + right_length);
        for (uint64_t i = 0; i < right_length; ++i)
            relator.push_back(row[used - 1 - i] ^ 1);
        relators.push_back(std::move(relator));
    }
    return WordsResult::success(std::move(relators));
}

struct Setup {
    uint64_t generators, bound;
    std::vector<std::vector<Entry>> fixed, subgroup;
};

Result<Setup> setup(const Request &req, bool with_subgroup) {
    using S = Result<Setup>;
    const Family &family = *req.family;
    if (family.kind != Family::Kind::Words)
        return S::failure(INVALID, "coset enumeration is defined on `words` families only");
    uint64_t generators = req.int_args.at("generators"), bound = req.int_args.at("max_cosets");
    if (generators == 0 || generators >= (1ULL << 31))
        return S::failure(INVALID, "generators must satisfy 1 <= generators < 2^31");
    if (family.p != 2 * generators)
        return S::failure(INVALID, "words alphabet must equal 2*generators");
    if (bound == 0 || bound >= (1ULL << 32))
        return S::failure(INVALID, "max_cosets must satisfy 1 <= max_cosets < 2^32");
    auto fixed = relation_relators_arg(req, 2 * generators);
    if (!fixed.ok) return S::failure(fixed.error.status, fixed.error.message);
    std::vector<std::vector<Entry>> subgroup;
    if (with_subgroup) {
        auto sub = subgroup_words_arg(req, 2 * generators);
        if (!sub.ok) return S::failure(sub.error.status, sub.error.message);
        subgroup = std::move(sub.value);
    }
    return S::success(Setup{generators, bound, std::move(fixed.value), std::move(subgroup)});
}

Answer answer_member(const Family &family, uint64_t index, const Setup &s, Matrix &member) {
    auto status = family.member_into(index, member);
    if (!status.ok) return {};
    std::vector<std::vector<Entry>> relators = s.fixed;
    relators.push_back(member.entries);
    return Table(s.generators, s.bound).enumerate(relators, s.subgroup);
}

R run_scalar(const Request &req, bool finite) {
    auto s = setup(req, !finite);
    if (!s.ok) return R::failure(s.error.status, s.error.message);
    auto size_r = req.family->size();
    if (!size_r.ok) return R::failure(size_r.error.status, size_r.error.message);
    uint64_t size = size_r.value;
    Reduction reduction = parse_reduction(req.reduction);
    Shared shared;
    auto prepared = prepare_all(reduction, size, shared);
    if (!prepared.ok) return R::failure(prepared.error.status, prepared.error.message);
    uint32_t threads = std::max<uint32_t>(1, std::min<uint64_t>(req.threads, size ? size : 1));
    std::vector<Accumulator> accumulators;
    for (uint32_t t = 0; t < threads; ++t) accumulators.emplace_back(reduction, &shared);
    auto statuses = parallel_ranges(size, threads, [&](uint32_t thread, uint64_t begin, uint64_t end) -> Status {
        Matrix member;
        for (uint64_t i = begin; i < end; ++i) {
            if (accumulators[thread].exhausted(i)) break;
            Answer answer = answer_member(*req.family, i, s.value, member);
            if (finite) accumulators[thread].boolean(i, answer.degree != 0);
            else accumulators[thread].integer(i, answer.degree);
        }
        return ok();
    });
    for (const auto &status : statuses)
        if (!status.ok) return R::failure(status.error.status, status.error.message);
    return assemble(req, reduction, accumulators, shared);
}

R run_representations(const Request &req) {
    auto s = setup(req, true);
    if (!s.ok) return R::failure(s.error.status, s.error.message);
    auto size_r = req.family->size();
    if (!size_r.ok) return R::failure(size_r.error.status, size_r.error.message);
    uint64_t size = size_r.value;
    auto out = std::make_shared<CosetRepresentations>();
    out->count = size; out->generators = s.value.generators; out->max_cosets = s.value.bound;
    out->degrees.assign(size, 0);
    unsigned __int128 entries = (unsigned __int128)size * s.value.generators * s.value.bound;
    if (entries > (1ULL << 40)) return R::failure(INVALID, "permutation representation output is too large to materialise");
    out->images.assign((uint64_t)entries, 0);
    uint32_t threads = std::max<uint32_t>(1, std::min<uint64_t>(req.threads, size ? size : 1));
    auto statuses = parallel_ranges(size, threads, [&](uint32_t, uint64_t begin, uint64_t end) -> Status {
        Matrix member;
        for (uint64_t i = begin; i < end; ++i) {
            Answer answer = answer_member(*req.family, i, s.value, member);
            out->degrees[i] = answer.degree;
            if (!answer.degree) continue;
            for (uint64_t g = 0; g < s.value.generators; ++g) {
                uint64_t dst = (i * s.value.generators + g) * s.value.bound;
                std::copy(answer.images.begin() + g * answer.degree,
                          answer.images.begin() + (g + 1) * answer.degree,
                          out->images.begin() + dst);
            }
        }
        return ok();
    });
    for (const auto &status : statuses)
        if (!status.ok) return R::failure(status.error.status, status.error.message);
    auto object = std::make_shared<Object>();
    object->kind = "coset_enumeration.representations";
    object->coset_representations = out;
    return R::success(object);
}

R run(const Request &req) {
    if (req.op == "index") return run_scalar(req, false);
    if (req.op == "is_finite") return run_scalar(req, true);
    if (req.op == "permutation_representation") return run_representations(req);
    return R::failure(4, "unknown coset_enumeration operation " + req.op);
}

BackendRegistration registration{Backend{
    "coset_enumeration", "generic",
    [] { return true; },
    [](const Request &) { return true; },
    run,
    0}};

} // namespace
} // namespace lk::coset_enumeration
