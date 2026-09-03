#include "../../../../runtime/src/reduce.hpp"

#include <boost/multiprecision/cpp_int.hpp>
#include <functional>
#include <limits>
#include <unordered_map>

namespace lk::young {
namespace {

using R = Result<std::shared_ptr<Object>>;
constexpr int INVALID = 1;

Result<std::vector<uint64_t>> natural_vector(const Request &req, const char *name) {
    auto it = req.handle_args.find(name);
    if (it == req.handle_args.end() || !it->second->matrix)
        return Result<std::vector<uint64_t>>::failure(INVALID, std::string("`") + name + "` must be an lk.naturals vector");
    const Matrix &m = *it->second->matrix;
    if (m.p != NATURALS || m.count != 1 || m.rows != 1)
        return Result<std::vector<uint64_t>>::failure(INVALID, std::string("`") + name + "` must be one row of natural numbers");
    return Result<std::vector<uint64_t>>::success(std::vector<uint64_t>(m.entries.begin(), m.entries.end()));
}

std::vector<uint64_t> shape_of(const Matrix &member) {
    std::vector<uint64_t> shape(member.entries.begin(), member.entries.end());
    while (!shape.empty() && shape.back() == 0) shape.pop_back();
    return shape;
}

uint64_t shape_size(const std::vector<uint64_t> &shape) {
    uint64_t n = 0;
    for (uint64_t row : shape) n += row;
    return n;
}

Result<uint64_t> hook_count(const std::vector<uint64_t> &shape) {
    using boost::multiprecision::cpp_int;
    uint64_t n = shape_size(shape);
    cpp_int value = 1;
    for (uint64_t k = 2; k <= n; ++k) value *= k;
    for (uint64_t i = 0; i < shape.size(); ++i) {
        for (uint64_t j = 0; j < shape[i]; ++j) {
            uint64_t below = 0;
            for (uint64_t r = i + 1; r < shape.size(); ++r) below += shape[r] > j;
            value /= shape[i] - j + below;
        }
    }
    if (value > UINT64_MAX) return Result<uint64_t>::failure(INVALID, "hook-length count does not fit in u64");
    return Result<uint64_t>::success(value.convert_to<uint64_t>());
}

Result<uint64_t> kostka(const std::vector<uint64_t> &shape, const std::vector<uint64_t> &weight) {
    std::vector<std::pair<uint64_t, uint64_t>> cells;
    for (uint64_t i = 0; i < shape.size(); ++i)
        for (uint64_t j = 0; j < shape[i]; ++j) cells.push_back({i, j});
    uint64_t cols = shape.empty() ? 0 : shape[0];
    std::vector<int64_t> filling(shape.size() * cols, -1);
    std::vector<uint64_t> remaining = weight;
    bool overflow = false;
    std::function<uint64_t(uint64_t)> count = [&](uint64_t pos) -> uint64_t {
        if (pos == cells.size()) return 1;
        auto [i, j] = cells[pos];
        uint64_t lower = 0;
        if (j) lower = (uint64_t)filling[i * cols + j - 1];
        if (i && j < shape[i - 1]) lower = std::max(lower, (uint64_t)filling[(i - 1) * cols + j] + 1);
        unsigned __int128 total = 0;
        for (uint64_t x = lower; x < remaining.size(); ++x) {
            if (!remaining[x]) continue;
            --remaining[x];
            filling[i * cols + j] = (int64_t)x;
            total += count(pos + 1);
            filling[i * cols + j] = -1;
            ++remaining[x];
            if (total > UINT64_MAX) { overflow = true; return 0; }
        }
        return (uint64_t)total;
    };
    uint64_t value = count(0);
    if (overflow) return Result<uint64_t>::failure(INVALID, "Kostka number does not fit in u64");
    return Result<uint64_t>::success(value);
}

bool adjacent(const std::pair<uint64_t, uint64_t> &a, const std::pair<uint64_t, uint64_t> &b) {
    return (a.first == b.first && (a.second + 1 == b.second || b.second + 1 == a.second)) ||
           (a.second == b.second && (a.first + 1 == b.first || b.first + 1 == a.first));
}

bool border_strip(const std::vector<uint64_t> &outer, const std::vector<uint64_t> &inner,
                  uint64_t length, uint64_t &height) {
    std::vector<std::pair<uint64_t, uint64_t>> removed;
    std::vector<bool> used_row(outer.size(), false);
    uint64_t cols = outer.empty() ? 0 : outer[0];
    std::vector<uint8_t> grid(outer.size() * cols, 0);
    for (uint64_t i = 0; i < outer.size(); ++i) {
        for (uint64_t j = inner[i]; j < outer[i]; ++j) {
            removed.push_back({i, j});
            used_row[i] = true;
            grid[i * cols + j] = 1;
        }
    }
    if (removed.size() != length || removed.empty()) return false;
    for (auto [i, j] : removed) {
        if (i + 1 < outer.size() && j + 1 < cols && grid[i * cols + j + 1] &&
            grid[(i + 1) * cols + j] && grid[(i + 1) * cols + j + 1]) return false;
    }
    std::vector<uint8_t> seen(removed.size(), 0);
    std::vector<uint64_t> queue{0};
    seen[0] = 1;
    for (uint64_t front = 0; front < queue.size(); ++front) {
        for (uint64_t i = 0; i < removed.size(); ++i) {
            if (!seen[i] && adjacent(removed[queue[front]], removed[i])) {
                seen[i] = 1;
                queue.push_back(i);
            }
        }
    }
    if (queue.size() != removed.size()) return false;
    height = 0;
    for (bool used : used_row) height += used;
    return true;
}

struct CharacterCalculator {
    const std::vector<uint64_t> &cycles;
    std::unordered_map<std::string, __int128> memo;

    std::string key(const std::vector<uint64_t> &shape, uint64_t pos) {
        std::string out = std::to_string(pos) + ":";
        for (uint64_t x : shape) out += std::to_string(x) + ",";
        return out;
    }

    __int128 value(const std::vector<uint64_t> &shape, uint64_t pos) {
        if (pos == cycles.size()) return shape_size(shape) == 0;
        std::string k = key(shape, pos);
        auto old = memo.find(k);
        if (old != memo.end()) return old->second;
        uint64_t strip = cycles[pos], total = shape_size(shape);
        if (strip > total) return 0;
        uint64_t target = total - strip;
        std::vector<uint64_t> inner(shape.size(), 0);
        __int128 answer = 0;
        std::function<void(uint64_t, uint64_t, uint64_t)> enumerate =
            [&](uint64_t row, uint64_t bound, uint64_t remaining) {
                if (row == shape.size()) {
                    if (remaining) return;
                    uint64_t height = 0;
                    if (border_strip(shape, inner, strip, height)) {
                        __int128 term = value(inner, pos + 1);
                        answer += (height % 2 ? term : -term);
                    }
                    return;
                }
                uint64_t largest = std::min({shape[row], bound, remaining});
                for (uint64_t width = largest;; --width) {
                    inner[row] = width;
                    enumerate(row + 1, width, remaining - width);
                    if (width == 0) break;
                }
            };
        enumerate(0, shape.empty() ? 0 : shape[0], target);
        memo.emplace(std::move(k), answer);
        return answer;
    }
};

template <class Fn>
R run_integer(const Request &req, Fn fn) {
    const Family &family = *req.family;
    auto size_result = family.size();
    if (!size_result.ok) return R::failure(size_result.error.status, size_result.error.message);
    uint64_t size = size_result.value;
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
            auto loaded = family.member_into(i, member);
            if (!loaded.ok) return loaded;
            auto value = fn(member);
            if (!value.ok) return fail(value.error.status, value.error.message);
            accumulators[thread].integer(i, value.value);
        }
        return ok();
    });
    for (const auto &status : statuses)
        if (!status.ok) return R::failure(status.error.status, status.error.message);
    return assemble(req, reduction, accumulators, shared);
}

R run_hook_length(const Request &req) {
    if (req.family->kind == Family::Kind::StandardTableaux) {
        std::vector<uint64_t> shape(req.family->data->entries.begin(), req.family->data->entries.end());
        auto value = hook_count(shape);
        if (!value.ok) return R::failure(value.error.status, value.error.message);
        return run_integer(req, [value](const Matrix &) { return value; });
    }
    return run_integer(req, [](const Matrix &member) { return hook_count(shape_of(member)); });
}

R run_kostka(const Request &req) {
    auto weight = natural_vector(req, "weight");
    if (!weight.ok) return R::failure(weight.error.status, weight.error.message);
    uint64_t sum = 0;
    for (uint64_t x : weight.value) sum += x;
    if (weight.value.empty() || sum != req.family->n)
        return R::failure(INVALID, "weight must sum to the partition size");
    return run_integer(req, [&](const Matrix &member) { return kostka(shape_of(member), weight.value); });
}

R run_character(const Request &req) {
    auto cycles = natural_vector(req, "cycle_type");
    if (!cycles.ok) return R::failure(cycles.error.status, cycles.error.message);
    uint64_t sum = 0, previous = UINT64_MAX;
    bool valid = !cycles.value.empty();
    for (uint64_t x : cycles.value) {
        valid &= x > 0 && x <= previous;
        previous = x;
        sum += x;
    }
    if (!valid || sum != req.family->n)
        return R::failure(INVALID, "cycle_type must be a positive weakly decreasing partition of n");
    if (req.reduction != "all") return R::failure(INVALID, "young.characters values only reduce with `all`");
    auto size_result = req.family->size();
    if (!size_result.ok) return R::failure(size_result.error.status, size_result.error.message);
    auto values = std::make_shared<Characters>();
    values->values.resize(size_result.value);
    auto statuses = parallel_ranges(size_result.value, req.threads, [&](uint32_t, uint64_t begin, uint64_t end) -> Status {
        Matrix member;
        for (uint64_t i = begin; i < end; ++i) {
            auto loaded = req.family->member_into(i, member);
            if (!loaded.ok) return loaded;
            CharacterCalculator calculator{cycles.value, {}};
            __int128 character = calculator.value(shape_of(member), 0);
            if (character < std::numeric_limits<int64_t>::min() || character > std::numeric_limits<int64_t>::max())
                return fail(INVALID, "character value does not fit in i64");
            values->values[i] = (int64_t)character;
        }
        return ok();
    });
    for (const auto &status : statuses)
        if (!status.ok) return R::failure(status.error.status, status.error.message);
    auto out = std::make_shared<Object>();
    out->kind = "young.characters";
    out->characters = values;
    return R::success(out);
}

R run_rsk(const Request &req) {
    if (req.reduction != "all") return R::failure(INVALID, "young.rsk_pairs values only reduce with `all`");
    auto size_result = req.family->size();
    if (!size_result.ok) return R::failure(size_result.error.status, size_result.error.message);
    uint64_t count = size_result.value, n = req.family->n;
    unsigned __int128 total_entries = (unsigned __int128)count * n * (1 + 2 * (unsigned __int128)n);
    if (total_entries > (1ULL << 31)) return R::failure(INVALID, "RSK output has more than 2^31 entries");
    auto pairs = std::make_shared<RskPairs>();
    pairs->count = count;
    pairs->length = n;
    pairs->shapes.assign(count * n, 0);
    pairs->insertion.assign(count * n * n, 0);
    pairs->recording.assign(count * n * n, 0);
    auto statuses = parallel_ranges(count, req.threads, [&](uint32_t, uint64_t begin, uint64_t end) -> Status {
        Matrix member;
        for (uint64_t index = begin; index < end; ++index) {
            auto loaded = req.family->member_into(index, member);
            if (!loaded.ok) return loaded;
            std::vector<std::vector<Entry>> insertion, recording;
            for (uint64_t step = 0; step < n; ++step) {
                Entry carry = member.entries[step];
                uint64_t row = 0;
                for (;; ++row) {
                    if (row == insertion.size()) {
                        insertion.push_back({carry});
                        recording.push_back({(Entry)(step + 1)});
                        break;
                    }
                    auto bumped = std::upper_bound(insertion[row].begin(), insertion[row].end(), carry);
                    if (bumped == insertion[row].end()) {
                        insertion[row].push_back(carry);
                        recording[row].push_back((Entry)(step + 1));
                        break;
                    }
                    std::swap(carry, *bumped);
                }
            }
            for (uint64_t row = 0; row < insertion.size(); ++row) {
                pairs->shapes[index * n + row] = (Entry)insertion[row].size();
                std::copy(insertion[row].begin(), insertion[row].end(),
                          pairs->insertion.begin() + (index * n + row) * n);
                std::copy(recording[row].begin(), recording[row].end(),
                          pairs->recording.begin() + (index * n + row) * n);
            }
        }
        return ok();
    });
    for (const auto &status : statuses)
        if (!status.ok) return R::failure(status.error.status, status.error.message);
    auto out = std::make_shared<Object>();
    out->kind = "young.rsk_pairs";
    out->rsk_pairs = pairs;
    return R::success(out);
}

R run(const Request &req) {
    if (req.op == "hook_length_count") return run_hook_length(req);
    if (req.op == "kostka") return run_kostka(req);
    if (req.op == "murnaghan_nakayama") return run_character(req);
    if (req.op == "rsk") return run_rsk(req);
    return R::failure(4, "unknown young operation " + req.op);
}

bool accepts(const Request &req) {
    if (req.family->kind == Family::Kind::Partitions) return req.family->n <= 30;
    if (req.op == "hook_length_count" && req.family->kind == Family::Kind::StandardTableaux) return true;
    if (req.op == "rsk" && req.family->kind == Family::Kind::Words) {
        auto size = req.family->size();
        if (!size.ok) return false;
        unsigned __int128 entries = (unsigned __int128)size.value * req.family->n *
                                    (1 + 2 * (unsigned __int128)req.family->n);
        return entries <= (1ULL << 31);
    }
    return false;
}

BackendRegistration registration{Backend{"young", "generic", [] { return true; }, accepts, run, 0}};

} // namespace
} // namespace lk::young
