// Read-only Action metadata adapter. The original window_segments algorithm
// and logical page identities are unchanged; no per-page Python tuple is made.
#pragma once

#include "qw3/kvmem_store.hpp"
#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace kvmem_action {

struct WindowSegments {
    std::vector<int32_t> starts, counts;
    std::vector<uint64_t> history;
};

// The caller retains the request's producer role. This read-only operation
// holds the GIL throughout; it neither changes residency nor borrows pointers
// into Store beyond the call. The Python arrays returned by the binding own
// their storage and survive subsequent append/selection/Store destruction.
inline WindowSegments window_segments(const qw3::KvMemStore &store,
        int64_t history_begin, int64_t local_begin, int64_t count,
        int64_t block_tokens) {
    if (history_begin < 0 || local_begin < 0 || count < 0 || block_tokens <= 0)
        throw std::invalid_argument("Invalid action window coordinates");
    if (local_begin > std::numeric_limits<int32_t>::max() ||
            count > std::numeric_limits<int32_t>::max() - local_begin)
        throw std::invalid_argument("Action window exceeds int32 kernel coordinates");

    struct Record {
        uint64_t begin;
        uint32_t count;
        int64_t position;
    };
    std::vector<Record> records;
    for (const auto &block : store.blocks()) {
        if (block.repository_generation && block.n_tokens &&
                (block.in_working_set || block.tier == qw3::KvTier::GPU))
            records.push_back({block.history_pos_start, block.n_tokens, block.baked_pos});
    }
    std::sort(records.begin(), records.end(), [](const auto &a, const auto &b) {
        return a.position < b.position;
    });
    WindowSegments result;
    int64_t cursor = 0;
    for (const auto &record : records) {
        if (record.position >= local_begin) break;
        // Check the gap before subtracting a potentially invalid baked_pos.
        // The old helper rejects the same row without publishing any result.
        if (record.position != cursor)
            throw std::invalid_argument("Action window has a gap or stale dense mapping");
        const int64_t n = std::min<int64_t>(record.count, local_begin-record.position);
        if (n <= 0 || n > block_tokens)
            throw std::invalid_argument("Action window has a gap or stale dense mapping");
        result.starts.push_back(static_cast<int32_t>(cursor));
        result.counts.push_back(static_cast<int32_t>(n));
        result.history.push_back(record.begin-record.begin%static_cast<uint64_t>(block_tokens));
        cursor += n;
    }
    // cursor <= local_begin, so this subtraction cannot overflow int64.
    int64_t logical = history_begin-(local_begin-cursor);
    if (logical < 0)
        throw std::invalid_argument("Invalid action suffix origin");
    const int64_t stop = local_begin+count;
    while (cursor < stop) {
        const int64_t n = std::min(block_tokens-logical%block_tokens, stop-cursor);
        result.starts.push_back(static_cast<int32_t>(cursor));
        result.counts.push_back(static_cast<int32_t>(n));
        result.history.push_back(static_cast<uint64_t>(logical-logical%block_tokens));
        cursor += n;
        // The final segment does not consume its exclusive logical end. Match
        // Python at INT64_MAX without rejecting an unused next coordinate.
        if (cursor < stop) {
            if (logical > std::numeric_limits<int64_t>::max()-n)
                throw std::invalid_argument("Action history exceeds int64 coordinates");
            logical += n;
        }
    }
    return result;
}

} // namespace kvmem_action
