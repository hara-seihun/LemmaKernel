/* Portable backend for statistics on partition and composition families. */
#include "../../../../runtime/src/reduce.hpp"

namespace lk::integer_partitions {
namespace {

using R = Result<std::shared_ptr<Object>>;

struct Walker : Family::Visitor {
    const Family &family;
    std::string op;
    std::vector<Entry> row;
    Accumulator acc;

    Walker(const Family &f, std::string operation, Reduction reduction, Shared *shared)
        : family(f), op(std::move(operation)), acc(reduction, shared) {}

    Step push(const Entry *values, Index first, Index) override {
        if (acc.exhausted(first)) return Step::Skip;
        row.assign(values, values + family.cols());
        return Step::Descend;
    }

    void pop() override {}

    void leaf(Index index) override {
        uint64_t count = 0;
        while (count < row.size() && row[count] != 0) ++count;
        uint64_t largest = count ? row[0] : 0;
        uint64_t value = 0;
        if (op == "number_of_parts") {
            value = count;
        } else if (op == "largest_part") {
            value = largest;
        } else if (op == "rank") {
            value = family.n + largest - count;
        } else {
            uint64_t ones = 0;
            for (uint64_t i = 0; i < count; ++i) ones += row[i] == 1;
            if (ones == 0) value = family.n + largest;
            else {
                uint64_t greater = 0;
                for (uint64_t i = 0; i < count; ++i) greater += row[i] > ones;
                value = family.n + greater - ones;
            }
        }
        acc.integer(index, value);
    }

    void take_all(Index, Index) override {}
    void skip_all(Index, Index) override {}
};

R run(const Request &req) {
    if (req.family->kind != Family::Kind::Partitions && req.family->kind != Family::Kind::Compositions)
        return R::failure(1, "integer partition operations need partitions or compositions families");
    if ((req.op == "rank" || req.op == "crank") && req.family->kind != Family::Kind::Partitions)
        return R::failure(1, req.op + " is defined on partitions families only");
    if (req.op != "number_of_parts" && req.op != "largest_part" && req.op != "rank" && req.op != "crank")
        return R::failure(4, "unknown integer_partitions operation " + req.op);

    auto size = req.family->size();
    if (!size.ok) return R::failure(size.error.status, size.error.message);
    auto tops = req.family->top_count();
    if (!tops.ok) return R::failure(tops.error.status, tops.error.message);
    Reduction reduction = parse_reduction(req.reduction);
    Shared shared;
    auto prepared = prepare_all(reduction, size.value, shared);
    if (!prepared.ok) return R::failure(prepared.error.status, prepared.error.message);

    uint32_t threads = std::max<uint32_t>(1, std::min<uint64_t>(req.threads, tops.value));
    std::vector<Walker> walkers;
    walkers.reserve(threads);
    for (uint32_t t = 0; t < threads; ++t) walkers.emplace_back(*req.family, req.op, reduction, &shared);
    auto statuses = parallel_ranges(tops.value, threads, [&](uint32_t t, uint64_t begin, uint64_t end) {
        return req.family->enumerate(walkers[t], begin, end);
    });
    for (const auto &status : statuses)
        if (!status.ok) return R::failure(status.error.status, status.error.message);
    std::vector<Accumulator> accs;
    for (auto &walker : walkers) accs.push_back(std::move(walker.acc));
    return assemble(req, reduction, accs, shared);
}

BackendRegistration registration{Backend{
    "integer_partitions", "generic",
    [] { return true; },
    [](const Request &req) {
        return req.family->kind == Family::Kind::Partitions || req.family->kind == Family::Kind::Compositions;
    },
    run,
    0}};

} // namespace
} // namespace lk::integer_partitions
