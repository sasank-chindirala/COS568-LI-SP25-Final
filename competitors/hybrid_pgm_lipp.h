#pragma once

#include "base.h"
#include "dynamic_pgm_index.h"
#include "lipp.h"
#include <memory>
#include <atomic>
#include <thread>
#include <mutex>

// True hybrid with adaptive flushing and no state discard

template<class KeyType, class SearchClass, size_t pgm_error>
class HybridPGMLIPP : public Competitor<KeyType, SearchClass> {
public:
    HybridPGMLIPP(const std::vector<int>& params)
        : dp_index_(params), lipp_index_(params), insert_count_(0),
          flush_threshold_(1000000), flushing_(false), total_ops_(0), insert_ops_(0) {}

    ~HybridPGMLIPP() {
        if (flush_thread_.joinable()) flush_thread_.join();
    }

    uint64_t Build(const std::vector<KeyValue<KeyType>>& data, size_t num_threads) {
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
        {
            std::lock_guard<std::mutex> lock(buffer_mutex_);
            dp_index_.Insert(data, thread_id);
            insert_buffer_.emplace_back(data);
            insert_count_++;
        }

        insert_ops_++;
        total_ops_++;

        if (total_ops_ == 100000 && flush_threshold_ == 1000000) {
            double ratio = static_cast<double>(insert_ops_) / total_ops_;
            flush_threshold_ = (ratio >= 0.5) ? 1000000 : 100000;  // adaptive tuning
        }

        if (insert_count_ >= flush_threshold_ && !flushing_) {
            std::lock_guard<std::mutex> lock(buffer_mutex_);
            flush_buffer_.swap(insert_buffer_);
            insert_count_ = 0;
            flushing_ = true;

            flush_thread_ = std::thread([this, thread_id]() {
                for (const auto& kv : flush_buffer_) {
                    lipp_index_.Insert(kv, thread_id);
                }
                flush_buffer_.clear();
                flushing_ = false;
            });
        }
    }

    std::string name() const {
        return "HybridPGMLIPP";
    }

    std::vector<std::string> variants() const {
        return { SearchClass::name(), std::to_string(pgm_error) };
    }

    size_t size() const {
        return dp_index_.size() + lipp_index_.size();
    }

    bool applicable(bool unique, bool range_query, bool insert, bool multithread,
                    const std::string& ops_filename) const {
        return !multithread;
    }

private:
    DynamicPGM<KeyType, SearchClass, pgm_error> dp_index_;
    Lipp<KeyType> lipp_index_;
    std::vector<KeyValue<KeyType>> insert_buffer_;
    std::vector<KeyValue<KeyType>> flush_buffer_;
    std::thread flush_thread_;
    std::mutex buffer_mutex_;
    std::atomic<bool> flushing_;
    size_t insert_count_;
    size_t flush_threshold_;
    std::atomic<size_t> total_ops_;
    std::atomic<size_t> insert_ops_;
};
