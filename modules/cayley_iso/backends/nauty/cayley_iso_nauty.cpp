#include "../cayley_iso_common.hpp"

#ifdef LEMMAKERNEL_HAVE_NAUTY
#include <nauty.h>
#endif

#include <mutex>
#include <numeric>

namespace lk::cayley_iso {
namespace {

#ifdef LEMMAKERNEL_HAVE_NAUTY
Result<std::vector<Entry>> canonical_graph(const GroupModel &group,
                                           const std::vector<uint64_t> &atom_words) {
    if (group.order > INT_MAX)
        return Result<std::vector<Entry>>::failure(INVALID, "nauty graph order exceeds int range");
    int n = (int)group.order;
    int words = SETWORDSNEEDED(n);
    std::vector<uint8_t> selected = selected_elements(group, atom_words);
    std::vector<::graph> source((size_t)words * n), canonical(source.size());
    EMPTYGRAPH(source.data(), words, n);
    for (int x = 0; x < n; ++x)
        for (int y = x + 1; y < n; ++y)
            if (selected[group.mul(group.inverse[x], y)])
                ADDONEEDGE(source.data(), x, y, words);

    std::vector<int> lab(n), partition(n), orbits(n);
    std::iota(lab.begin(), lab.end(), 0);
    std::fill(partition.begin(), partition.end(), 1);
    if (!partition.empty()) partition.back() = 0;
    DEFAULTOPTIONS_GRAPH(options);
    options.getcanon = TRUE;
    options.defaultptn = TRUE;
    options.writeautoms = FALSE;
    options.outfile = nullptr;
    statsblk stats{};
    auto canonicalize = [&] {
        nauty_check(WORDSIZE, words, n, NAUTYVERSIONID);
        densenauty(source.data(), lab.data(), partition.data(), orbits.data(),
                   &options, &stats, words, n, canonical.data());
    };
#ifdef LEMMAKERNEL_SERIALIZE_NAUTY
    static std::mutex nauty_mutex;
    {
        std::lock_guard lock(nauty_mutex);
        canonicalize();
    }
#else
    canonicalize();
#endif
    if (stats.errstatus != 0)
        return Result<std::vector<Entry>>::failure(INTERNAL, "nauty canonicalization failed");

    std::vector<Entry> key((size_t)n * n, 0);
    for (int x = 0; x < n; ++x) {
        const setword *row = GRAPHROW(canonical.data(), x, words);
        for (int y = 0; y < n; ++y) key[(size_t)x * n + y] = ISELEMENT(row, y) ? 1 : 0;
    }
    return Result<std::vector<Entry>>::success(std::move(key));
}
#endif

BackendRegistration registration{Backend{
    "cayley_iso", "nauty",
    [] {
#ifdef LEMMAKERNEL_HAVE_NAUTY
        return true;
#else
        return false;
#endif
    },
    [](const Request &req) { return req.family->kind == Family::Kind::GroupTables; },
    [](const Request &req) -> R {
#ifdef LEMMAKERNEL_HAVE_NAUTY
        return run_backend(req, canonical_graph);
#else
        return R::failure(INVALID, "cayley_iso.nauty was built without TLS-enabled nauty");
#endif
    },
    100}};

} // namespace
} // namespace lk::cayley_iso
