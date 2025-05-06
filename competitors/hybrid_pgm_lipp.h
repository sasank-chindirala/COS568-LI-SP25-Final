#pragma once

#include "base.h"
#include "dynamic_pgm_index.h"
#include "lipp.h"
#include <memory>

template<class KeyType, class SearchClass, size_t pgm_error>
class HybridPGMLIPP : public Competitor<KeyType, SearchClass> {
public:
    HybridPGMLIPP(const std::vector<int>& params)
        : dp_index_(params), lipp_index_(params), insert_count_(0), flush_threshold_(1000000) {} // threshold is tunable

    uint64_t Build(const std::vector<KeyValue<KeyType>>& data, size_t num_threads) {
        // Build only LIPP from the initial dataset
        return lipp_index_.Build(data, num_threads);
    }

    size_t EqualityLookup(const KeyType& lookup_key, uint32_t thread_id) const {
        size_t result = dp_index_.EqualityLookup(lookup_key, thread_id);
        if (result == util::OVERFLOW || result == util::NOT_FOUND) {
            return lipp_index_.EqualityLookup(lookup_key, thread_id);
        }
        return result;
    }

    uint64_t RangeQuery(const KeyType& lower_key, const KeyType& upper_key, uint32_t thread_id) const {
        return dp_index_.RangeQuery(lower_key, upper_key, thread_id) +
               lipp_index_.RangeQuery(lower_key, upper_key, thread_id);
    }

    void Insert(const KeyValue<KeyType>& data, uint32_t thread_id) {
        dp_index_.Insert(data, thread_id);
        insert_buffer_.emplace_back(data);
        insert_count_++;

        if (insert_count_ >= flush_threshold_) {
            flush_to_lipp(thread_id);
        }
    }

    std::string name() const {
        return "HybridPGMLIPP";
    }

    std::vector<std::string> variants() const {
        std::vector<std::string> vec;
        vec.push_back(SearchClass::name());
        vec.push_back(std::to_string(pgm_error));
        return vec;
    }

    size_t size() const {
        return dp_index_.size() + lipp_index_.size();
    }

    bool applicable(bool unique, bool range_query, bool insert, bool multithread,
                    const std::string& ops_filename) const {
        return !multithread;
    }

private:
    void flush_to_lipp(uint32_t thread_id) {
        for (const auto& kv : insert_buffer_) {
            lipp_index_.Insert(kv, thread_id);
        }
        insert_buffer_.clear();
        dp_index_ = DynamicPGM<KeyType, SearchClass, pgm_error>(std::vector<int>());
        insert_count_ = 0;
    }

    DynamicPGM<KeyType, SearchClass, pgm_error> dp_index_;
    Lipp<KeyType> lipp_index_;
    std::vector<KeyValue<KeyType>> insert_buffer_;
    size_t insert_count_;
    size_t flush_threshold_;
};
