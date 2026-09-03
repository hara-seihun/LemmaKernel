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
        bind(identity_, identity_);
        descend();
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

    void descend() {
        uint64_t source = order_;
        size_t fewest = SIZE_MAX;
        for (uint64_t a = 0; a < order_; ++a) {
            if (image_[a] >= 0) continue;
            size_t choices = 0;
            for (uint64_t b = 0; b < order_; ++b)
                if (preimage_[b] < 0 && compatible(a, b)) ++choices;
            if (choices < fewest) {
                fewest = choices;
                source = a;
            }
        }
        if (source == order_) {
            std::vector<Entry> automorphism(order_);
            for (uint64_t a = 0; a < order_; ++a) automorphism[a] = (Entry)image_[a];
            found_.push_back(std::move(automorphism));
            return;
        }
        for (uint64_t target = 0; target < order_; ++target) {
            if (preimage_[target] >= 0 || !compatible(source, target)) continue;
            size_t mark = trail_.size();
            if (bind(source, target)) descend();
            rollback(mark);
        }
    }
};

inline std::vector<std::vector<Entry>> automorphisms(const Entry *table, uint64_t order) {
    return AutomorphismSearch(table, order).run();
}

} // namespace lk::group_table
