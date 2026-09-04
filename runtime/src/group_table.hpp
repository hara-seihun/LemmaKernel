#pragma once

#include "object.hpp"

#include <algorithm>
#include <cstddef>

namespace lk::group_table {

/* Enumerate every automorphism of a validated Cayley table. A partial automorphism is an
 * injective map between generated subgroups. Each assigned image propagates through every
 * product whose other factor is already assigned. */
class AutomorphismSearch {
public:
    AutomorphismSearch(const Entry *table, uint64_t order)
        : table_(table), order_(order), image_(order, -1), preimage_(order, -1),
          element_order_(order), centralizer_size_(order) {
        identity_ = find_identity();
        for (uint64_t a = 0; a < order_; ++a) {
            element_order_[a] = order_of(a);
            uint64_t size = 0;
            for (uint64_t b = 0; b < order_; ++b) size += mul(a, b) == mul(b, a);
            centralizer_size_[a] = size;
        }
    }

    std::vector<std::vector<Entry>> run() {
        generators_ = generators();
        bind(identity_, identity_);
        descend(0);
        std::sort(found_.begin(), found_.end());
        return found_;
    }

private:
    const Entry *table_;
    uint64_t order_;
    uint64_t identity_ = 0;
    std::vector<int64_t> image_;
    std::vector<int64_t> preimage_;
    std::vector<uint64_t> element_order_;
    std::vector<uint64_t> centralizer_size_;
    std::vector<uint64_t> trail_;
    std::vector<uint64_t> generators_;
    std::vector<std::vector<Entry>> found_;

    Entry mul(uint64_t a, uint64_t b) const { return table_[a * order_ + b]; }

    uint64_t find_identity() const {
        for (uint64_t identity = 0; identity < order_; ++identity) {
            bool valid = true;
            for (uint64_t x = 0; x < order_; ++x)
                if (mul(identity, x) != x || mul(x, identity) != x) {
                    valid = false;
                    break;
                }
            if (valid) return identity;
        }
        return order_;
    }

    uint64_t order_of(uint64_t a) const {
        uint64_t x = identity_;
        for (uint64_t k = 1; k <= order_; ++k) {
            x = mul(x, a);
            if (x == identity_) return k;
        }
        return 0;
    }

    bool compatible(uint64_t source, uint64_t target) const {
        return element_order_[source] == element_order_[target] &&
               centralizer_size_[source] == centralizer_size_[target];
    }

    std::vector<uint8_t> closure(const std::vector<uint64_t> &generators) const {
        std::vector<uint8_t> reached(order_, 0);
        std::vector<uint64_t> queue{identity_};
        reached[identity_] = 1;
        for (size_t head = 0; head < queue.size(); ++head) {
            uint64_t x = queue[head];
            for (uint64_t generator : generators) {
                uint64_t products[2]{mul(x, generator), mul(generator, x)};
                for (uint64_t product : products)
                    if (!reached[product]) {
                        reached[product] = 1;
                        queue.push_back(product);
                    }
            }
        }
        return reached;
    }

    std::vector<uint64_t> generators() const {
        std::vector<uint64_t> result;
        auto reached = closure(result);
        while (std::count(reached.begin(), reached.end(), uint8_t{1}) < (ptrdiff_t)order_) {
            uint64_t best = order_;
            size_t best_growth = 0;
            size_t best_choices = SIZE_MAX;
            for (uint64_t candidate = 0; candidate < order_; ++candidate) {
                if (reached[candidate]) continue;
                auto trial = result;
                trial.push_back(candidate);
                auto expanded = closure(trial);
                size_t growth = std::count(expanded.begin(), expanded.end(), uint8_t{1});
                size_t choices = 0;
                for (uint64_t target = 0; target < order_; ++target)
                    choices += compatible(candidate, target);
                if (growth > best_growth ||
                    (growth == best_growth && choices < best_choices) ||
                    (growth == best_growth && choices == best_choices && candidate < best)) {
                    best = candidate;
                    best_growth = growth;
                    best_choices = choices;
                }
            }
            result.push_back(best);
            reached = closure(result);
        }
        return result;
    }

    bool bind(uint64_t source, uint64_t target) {
        if (image_[source] >= 0) return (uint64_t)image_[source] == target;
        if (preimage_[target] >= 0 || !compatible(source, target)) return false;
        image_[source] = (int64_t)target;
        preimage_[target] = (int64_t)source;
        trail_.push_back(source);

        for (uint64_t x = 0; x < order_; ++x) {
            if (image_[x] < 0) continue;
            uint64_t image = (uint64_t)image_[x];
            if (!bind(mul(source, x), mul(target, image))) return false;
            if (!bind(mul(x, source), mul(image, target))) return false;
        }
        return true;
    }

    void rollback(size_t mark) {
        while (trail_.size() > mark) {
            uint64_t source = trail_.back();
            trail_.pop_back();
            preimage_[(uint64_t)image_[source]] = -1;
            image_[source] = -1;
        }
    }

    void descend(size_t generator_index) {
        while (generator_index < generators_.size() && image_[generators_[generator_index]] >= 0)
            ++generator_index;
        if (generator_index == generators_.size()) {
            if (std::find(image_.begin(), image_.end(), -1) != image_.end()) return;
            std::vector<Entry> automorphism(order_);
            for (uint64_t a = 0; a < order_; ++a) automorphism[a] = (Entry)image_[a];
            found_.push_back(std::move(automorphism));
            return;
        }
        uint64_t source = generators_[generator_index];
        for (uint64_t target = 0; target < order_; ++target) {
            if (preimage_[target] >= 0 || !compatible(source, target)) continue;
            size_t mark = trail_.size();
            if (bind(source, target)) descend(generator_index + 1);
            rollback(mark);
        }
    }
};

inline std::vector<std::vector<Entry>> automorphisms(const Entry *table, uint64_t order) {
    return AutomorphismSearch(table, order).run();
}

} // namespace lk::group_table
