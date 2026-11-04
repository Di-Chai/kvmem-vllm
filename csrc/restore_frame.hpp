// Host-only adapter for RequestMemory.restore_window. Keep qw3's selector,
// residency and lease operations unchanged; remove per-page Python objects.
#pragma once

#include "qw3/kvmem_store.hpp"
#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <set>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

namespace kvmem_restore {

using FrameRecord = std::tuple<uint32_t, uint64_t, uint64_t, uint32_t, int64_t>;

// The caller owns the request's producer role and, after this returns, must
// retain the Plan and fence assembly before the existing commit_selection.
// This binding intentionally retains the GIL. Repository leases protect bytes,
// not arbitrary concurrent structural Store mutations or explicit retirement.
struct ValidatedFrame {
    std::vector<qw3::KvMemBlockHandle> saved;
    std::vector<qw3::KvMemStorageRead> reads;
};

// Read-only validation shared with capture. Keep read leases alive until the
// caller has finished; capture must never call restore/select to validate.
inline ValidatedFrame validate_frame(qw3::KvMemStore &store,
        const std::shared_ptr<qw3::KvMemStore::RepositoryCheckpoint> &prefix,
        const std::vector<FrameRecord> &records) {
    if (store.repository_transaction_pending())
        throw std::logic_error("KVMem frame restore requires no pending repository transaction");
    if (!store.repository_checkpoint_valid(prefix))
        throw std::invalid_argument("Stale or foreign window checkpoint");

    const auto &config = store.config();
    ValidatedFrame frame;
    auto &saved = frame.saved;
    auto &reads = frame.reads;
    saved.reserve(records.size());
    reads.reserve(records.size());
    std::set<std::pair<uint32_t, uint64_t>> seen;
    uint64_t dense = 0, logical_end = 0;
    for (const auto &[slot, generation, begin, count, position] : records) {
        if (begin > std::numeric_limits<uint64_t>::max() - count ||
                position < 0 || static_cast<uint64_t>(position) != dense ||
                begin < logical_end || begin % config.block_tokens || !count || count > config.block_tokens ||
                (count != config.block_tokens && begin + count != prefix->position()) ||
                begin + count > prefix->position() || !seen.emplace(slot, generation).second)
            throw std::invalid_argument("Checkpoint window is not a complete ordered dense frame");
        const qw3::KvMemBlockHandle handle{slot, generation};
        if (store.repository_handle(slot) != handle || !generation)
            throw std::invalid_argument("Checkpoint window contains a recycled generation");
        auto read = store.repository_storage_read(begin, count);
        if (read.spans.size() != 1 || read.spans[0].handle != handle ||
                read.spans[0].token_count != count)
            throw std::invalid_argument("Checkpoint window no longer matches immutable history");
        saved.push_back(handle);
        reads.push_back(std::move(read));
        dense += count;
        logical_end = begin + count;
    }
    const uint64_t limit = std::min(prefix->position(),
        static_cast<uint64_t>(config.select_budget) + config.gen_budget);
    if (!dense || dense > limit)
        throw std::invalid_argument("Checkpoint window exceeds its original bounded extent");
    return frame;
}

inline qw3::KvMemPlan restore_frame(qw3::KvMemStore &store,
        const std::shared_ptr<qw3::KvMemStore::RepositoryCheckpoint> &prefix,
        const std::vector<FrameRecord> &records) {
    auto frame = validate_frame(store, prefix, records);
    const auto &saved = frame.saved;
    if (store.history_end() != prefix->position())
        throw std::invalid_argument("Rewind authority before restoring its checkpoint window");

    // Admit only the saved frame. Store transfers already-owned current-window
    // leases through the plan, including permanent GC retirement, before cold
    // reset. The caller retains the plan through its assembly completion fence.
    return store.restore_repository_selection(saved);
}

} // namespace kvmem_restore
