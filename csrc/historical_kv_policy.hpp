// Host-only historical KV policy. This layer never pins, retires or frees KV.
#pragma once

#include "qw3/kvmem_repository.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cmath>
#include <stdexcept>
#include <tuple>
#include <vector>

namespace vllm_kvmem {

// A sole semantic producer publishes a whole, generation-qualified observation.
// Counters count real observation opportunities; elapsed GC polls are not evidence.
struct HistoricalEvidence {
    uint64_t revision = 0, observations = 0, cold_observations = 0;
    bool eligible = false;
    double importance = 1.0;
    // -1 means unknown. A configured model threshold never accepts unknown.
    double model_reclaim_score = -1.0;
};

struct HistoricalBlockMeta {
    qw3::KvMemBlockHandle handle;
    uint64_t logical_id = 0, birth_seq = 0, last_use_seq = 0, cpu_bytes = 0;
    bool live = false, sealed = false, mandatory = false;
    // Observations, not mechanism-level deletion vetoes.
    bool gpu_resident = false;
    uint64_t references = 0;
    uint32_t selection_holds = 0;
    HistoricalEvidence evidence;
};

struct HistoricalSnapshot {
    uint64_t incarnation = 0, round = 0;
    // Immutable while a policy round is in progress; owned by the manager.
    std::vector<HistoricalBlockMeta> blocks;
};

struct HistoricalPressure {
    uint32_t ready_slots = 0, low_slots = 0, high_slots = 0, pending_slots = 0;
    uint64_t free_cpu_bytes = 0, low_cpu_bytes = 0, high_cpu_bytes = 0;
    uint64_t pending_cpu_bytes = 0;
    uint32_t requested_ready_slots = 0;
    uint64_t scope_cpu_bytes = 0, scope_cpu_limit = UINT64_MAX, scope_pending_cpu_bytes = 0;
};

struct HistoricalWorkBudget {
    size_t scan_blocks = 256, delete_blocks = 64;
};

struct HistoricalDeleteIntent {
    qw3::KvMemBlockHandle handle;
    uint64_t incarnation = 0, round = 0;
    // Zero preserves custom/LRU policies. Statistical decisions require the
    // exact evidence revision to survive commit-time revalidation.
    uint64_t evidence_revision = 0;
};

enum class HistoricalPolicyStatus { Yielded, Complete, NoCandidate, Backpressure };
struct HistoricalPolicyResult {
    HistoricalPolicyStatus status;
    size_t examined = 0, emitted = 0;
};

// The manager retains/acknowledges every intent before clearing this batch.
// No allocation on push; constructor allocation belongs to initialization.
class HistoricalDecisionBatch {
public:
    explicit HistoricalDecisionBatch(size_t capacity) : capacity_(capacity) {
        if (!capacity) throw std::invalid_argument("Historical decision capacity must be positive");
        values_.reserve(capacity);
    }
    const std::vector<HistoricalDeleteIntent> &values() const { return values_; }
    size_t remaining() const { return capacity_ - values_.size(); }
    bool empty() const { return values_.empty(); }
    bool push(HistoricalDeleteIntent intent) {
        if (!remaining()) return false;
        values_.push_back(intent);
        return true;
    }
    void clear_after_ack() { values_.clear(); }
private:
    size_t capacity_;
    std::vector<HistoricalDeleteIntent> values_;
};

class HistoricalKvPolicy {
public:
    virtual ~HistoricalKvPolicy() = default;
    // Called by the backend before a preallocated scope is reused. Stateful
    // factory policies must clear round/candidate state without reallocating.
    virtual void reset() {}
    // Policy-owned trigger. Custom proactive policies default to requesting a
    // snapshot even without capacity pressure; the mechanism never vetoes it.
    virtual bool wants_snapshot(const HistoricalPressure &) { return true; }
    // Strategy-owned revalidation of a delayed intent. The mechanism does not
    // impose sealed/mandatory/GPU eligibility rules on custom strategies.
    virtual bool permits(const HistoricalBlockMeta &) const { return true; }
    virtual HistoricalPolicyResult step(const HistoricalSnapshot &, const HistoricalPressure &,
                                        HistoricalWorkBudget, HistoricalDecisionBatch &) = 0;
};

// Observed-snapshot LRU, not a claim of a simultaneous global access snapshot.
// Eligibility choices (sealed, mandatory) live here; the retirement mechanism
// accepts any published generation regardless of GPU residency or lease count.
class HistoricalLruPolicy final : public HistoricalKvPolicy {
public:
    explicit HistoricalLruPolicy(size_t capacity, bool keep_mandatory = true)
        : capacity_(capacity), keep_mandatory_(keep_mandatory) {
        if (!capacity) throw std::invalid_argument("Historical policy capacity must be positive");
        candidates_.reserve(capacity);
    }

    bool wants_snapshot(const HistoricalPressure &pressure) override {
        update_pressure(pressure);
        return (slots_pressure_ && deficit(pressure.high_slots, pressure.ready_slots,
                                           pressure.pending_slots)) ||
               (bytes_pressure_ && deficit(pressure.high_cpu_bytes, pressure.free_cpu_bytes,
                                           pressure.pending_cpu_bytes)) ||
               deficit(pressure.requested_ready_slots, pressure.ready_slots, pressure.pending_slots) ||
               deficit(pressure.scope_cpu_bytes, pressure.scope_cpu_limit, pressure.scope_pending_cpu_bytes);
    }
    bool permits(const HistoricalBlockMeta &block) const override {
        return block.live && block.sealed && (!keep_mandatory_ || !block.mandatory);
    }
    void reset() override {
        initialized_ = done_ = slots_pressure_ = bytes_pressure_ = false;
        cursor_ = 0;
        candidates_.clear();
    }

    HistoricalPolicyResult step(const HistoricalSnapshot &snapshot,
                                const HistoricalPressure &pressure,
                                HistoricalWorkBudget budget,
                                HistoricalDecisionBatch &out) override {
        validate(snapshot, pressure, budget);
        // The manager must commit or retain all old decisions, including Retry,
        // before requesting another batch. Never silently overwrite decisions.
        if (!out.empty()) return {HistoricalPolicyStatus::Backpressure, 0, 0};
        if (!initialized_ || incarnation_ != snapshot.incarnation || round_ != snapshot.round) {
            if (initialized_ && !done_)
                throw std::logic_error("Historical snapshot changed before policy round completion");
            if (!initialized_ || incarnation_ != snapshot.incarnation)
                slots_pressure_ = bytes_pressure_ = false;
            initialized_ = true;
            incarnation_ = snapshot.incarnation;
            round_ = snapshot.round;
            cursor_ = 0;
            done_ = false;
            terminal_ = HistoricalPolicyStatus::Complete;
            candidates_.clear();
        }
        if (done_) return {terminal_, 0, 0};
        update_pressure(pressure);
        // Pressure is refreshed per step, independently of the immutable LRU
        // snapshot. A returned request reservation can end pressure mid-scan.
        uint64_t slots = slots_pressure_ ? deficit(pressure.high_slots, pressure.ready_slots,
                                                   pressure.pending_slots) : 0;
        uint64_t bytes = bytes_pressure_ ? deficit(pressure.high_cpu_bytes, pressure.free_cpu_bytes,
                                                   pressure.pending_cpu_bytes) : 0;
        slots = std::max(slots, deficit(pressure.requested_ready_slots, pressure.ready_slots, pressure.pending_slots));
        bytes = std::max(bytes, deficit(pressure.scope_cpu_bytes, pressure.scope_cpu_limit,
                                       pressure.scope_pending_cpu_bytes));
        if (!slots && !bytes) return finish(HistoricalPolicyStatus::Complete, 0, 0);

        size_t examined = 0;
        while (cursor_ < snapshot.blocks.size() && examined < budget.scan_blocks) {
            const auto &block = snapshot.blocks[cursor_++];
            ++examined;
            if (!block.live || !block.sealed || !block.handle.generation ||
                    (keep_mandatory_ && block.mandatory)) continue;
            candidates_.push_back(block);
            // Building the heap incrementally avoids an uninterruptible full
            // sort after the final scan slice. Work is O(log N) per entry.
            std::push_heap(candidates_.begin(), candidates_.end(), Newer{});
        }
        if (cursor_ < snapshot.blocks.size())
            return {HistoricalPolicyStatus::Yielded, examined, 0};

        size_t emitted = 0;
        while (!candidates_.empty() && (slots || bytes) &&
                emitted < budget.delete_blocks && out.remaining()) {
            std::pop_heap(candidates_.begin(), candidates_.end(), Newer{});
            const auto block = candidates_.back();
            candidates_.pop_back();
            out.push({block.handle, snapshot.incarnation, snapshot.round});
            ++emitted;
            if (slots) --slots;
            bytes = bytes > block.cpu_bytes ? bytes - block.cpu_bytes : 0;
        }
        if (candidates_.empty())
            return finish((slots || bytes) ? HistoricalPolicyStatus::NoCandidate
                                          : HistoricalPolicyStatus::Complete, examined, emitted);
        // Preserve this heap across acknowledged batches. The manager supplies
        // updated pending credits on the next step; no repeated full scan.
        return {HistoricalPolicyStatus::Yielded, examined, emitted};
    }

private:
    struct Newer {
        static auto key(const HistoricalBlockMeta &b) {
            return std::make_tuple(b.last_use_seq, b.birth_seq, b.logical_id,
                                   b.handle.slot, b.handle.generation);
        }
        bool operator()(const HistoricalBlockMeta &a, const HistoricalBlockMeta &b) const {
            return key(a) > key(b);
        }
    };
    static uint64_t deficit(uint64_t target, uint64_t free, uint64_t pending) {
        const auto missing = target > free ? target - free : 0;
        return missing > pending ? missing - pending : 0;
    }
    void validate(const HistoricalSnapshot &s, const HistoricalPressure &p,
                  HistoricalWorkBudget b) const {
        if (s.blocks.size() > capacity_) throw std::length_error("Historical snapshot exceeds capacity");
        if (!b.scan_blocks || !b.delete_blocks)
            throw std::invalid_argument("Historical work budgets must be positive");
        if (p.requested_ready_slots > capacity_)
            throw std::invalid_argument("Requested historical reserve exceeds scope capacity");
        if ((p.low_slots || p.high_slots) &&
                (!p.low_slots || p.low_slots >= p.high_slots || p.high_slots > capacity_))
            throw std::invalid_argument("Invalid historical slot watermarks");
        if ((p.low_cpu_bytes || p.high_cpu_bytes) &&
                (!p.low_cpu_bytes || p.low_cpu_bytes >= p.high_cpu_bytes))
            throw std::invalid_argument("Invalid historical CPU byte watermarks");
    }
    void update_pressure(const HistoricalPressure &p) {
        if (!p.high_slots || p.ready_slots >= p.high_slots) slots_pressure_ = false;
        else if (p.ready_slots < p.low_slots) slots_pressure_ = true;
        if (!p.high_cpu_bytes || p.free_cpu_bytes >= p.high_cpu_bytes) bytes_pressure_ = false;
        else if (p.free_cpu_bytes < p.low_cpu_bytes) bytes_pressure_ = true;
    }
    HistoricalPolicyResult finish(HistoricalPolicyStatus status, size_t examined, size_t emitted) {
        done_ = true;
        terminal_ = status;
        return {status, examined, emitted};
    }
    size_t capacity_, cursor_ = 0;
    uint64_t incarnation_ = 0, round_ = 0;
    bool keep_mandatory_, initialized_ = false, done_ = false;
    bool slots_pressure_ = false, bytes_pressure_ = false;
    HistoricalPolicyStatus terminal_ = HistoricalPolicyStatus::Complete;
    std::vector<HistoricalBlockMeta> candidates_;
};

// Conservative threshold policy. Deterministic bounded scanning is deliberate:
// no KV scan, sorting, model callback, allocation or device work in a round.
class HistoricalStatisticalPolicy final : public HistoricalKvPolicy {
public:
    struct Config {
        uint64_t min_observations = 3, min_cold_observations = 3;
        double max_importance = 0.1, min_model_reclaim_score = -1.0;
        bool proactive = true;
    };
    explicit HistoricalStatisticalPolicy(size_t capacity)
        : HistoricalStatisticalPolicy(capacity, Config{}) {}
    HistoricalStatisticalPolicy(size_t capacity, Config config)
        : capacity_(capacity), config_(config) {
        if (!capacity || !config.min_observations ||
                !std::isfinite(config.max_importance) || config.max_importance < 0 || config.max_importance > 1 ||
                !std::isfinite(config.min_model_reclaim_score) ||
                (config.min_model_reclaim_score != -1 &&
                 (config.min_model_reclaim_score < 0 || config.min_model_reclaim_score > 1)))
            throw std::invalid_argument("Invalid statistical historical policy configuration");
    }
    void reset() override {
        initialized_ = done_ = slots_pressure_ = bytes_pressure_ = false;
        cursor_ = 0;
    }
    bool wants_snapshot(const HistoricalPressure &pressure) override {
        update_pressure(pressure);
        return config_.proactive || targets(pressure) != std::make_pair(uint64_t{0}, uint64_t{0});
    }
    bool permits(const HistoricalBlockMeta &block) const override {
        const auto &e = block.evidence;
        return block.live && block.sealed && !block.mandatory && !block.selection_holds &&
               block.handle.generation && e.revision && e.eligible &&
               e.observations >= config_.min_observations &&
               e.cold_observations >= config_.min_cold_observations &&
               e.cold_observations <= e.observations &&
               std::isfinite(e.importance) && e.importance >= 0 && e.importance <= config_.max_importance &&
               (config_.min_model_reclaim_score < 0 ||
                (std::isfinite(e.model_reclaim_score) && e.model_reclaim_score >= config_.min_model_reclaim_score &&
                 e.model_reclaim_score <= 1));
    }
    HistoricalPolicyResult step(const HistoricalSnapshot &snapshot, const HistoricalPressure &pressure,
                                HistoricalWorkBudget budget, HistoricalDecisionBatch &out) override {
        if (snapshot.blocks.size() > capacity_)
            throw std::length_error("Historical snapshot exceeds capacity");
        if (!budget.scan_blocks || !budget.delete_blocks)
            throw std::invalid_argument("Historical work budgets must be positive");
        if (pressure.requested_ready_slots > capacity_ ||
                ((pressure.low_slots || pressure.high_slots) &&
                 (!pressure.low_slots || pressure.low_slots >= pressure.high_slots || pressure.high_slots > capacity_)) ||
                ((pressure.low_cpu_bytes || pressure.high_cpu_bytes) &&
                 (!pressure.low_cpu_bytes || pressure.low_cpu_bytes >= pressure.high_cpu_bytes)))
            throw std::invalid_argument("Invalid historical pressure watermarks");
        if (!out.empty()) return {HistoricalPolicyStatus::Backpressure, 0, 0};
        if (!initialized_ || incarnation_ != snapshot.incarnation || round_ != snapshot.round) {
            if (initialized_ && !done_)
                throw std::logic_error("Historical snapshot changed before policy round completion");
            initialized_ = true;
            incarnation_ = snapshot.incarnation;
            round_ = snapshot.round;
            cursor_ = 0;
            done_ = false;
        }
        if (done_) return {HistoricalPolicyStatus::Complete, 0, 0};
        update_pressure(pressure);
        auto [slots, bytes] = targets(pressure);
        if (!config_.proactive && !slots && !bytes) {
            done_ = true;
            return {HistoricalPolicyStatus::Complete, 0, 0};
        }
        size_t examined = 0, emitted = 0;
        while (cursor_ < snapshot.blocks.size() && examined < budget.scan_blocks &&
                emitted < budget.delete_blocks && out.remaining() &&
                (config_.proactive || slots || bytes)) {
            const auto &block = snapshot.blocks[cursor_++];
            ++examined;
            if (!permits(block)) continue;
            out.push({block.handle, snapshot.incarnation, snapshot.round, block.evidence.revision});
            ++emitted;
            if (slots) --slots;
            bytes = bytes > block.cpu_bytes ? bytes - block.cpu_bytes : 0;
        }
        done_ = cursor_ == snapshot.blocks.size() || (!config_.proactive && !slots && !bytes);
        return {done_ ? ((!emitted || (!config_.proactive && (slots || bytes))) ?
                            HistoricalPolicyStatus::NoCandidate : HistoricalPolicyStatus::Complete)
                      : HistoricalPolicyStatus::Yielded,
                examined, emitted};
    }
private:
    static uint64_t deficit(uint64_t target, uint64_t available, uint64_t pending) {
        const auto missing = target > available ? target - available : 0;
        return missing > pending ? missing - pending : 0;
    }
    void update_pressure(const HistoricalPressure &p) {
        if (!p.high_slots || p.ready_slots >= p.high_slots) slots_pressure_ = false;
        else if (p.ready_slots < p.low_slots) slots_pressure_ = true;
        if (!p.high_cpu_bytes || p.free_cpu_bytes >= p.high_cpu_bytes) bytes_pressure_ = false;
        else if (p.free_cpu_bytes < p.low_cpu_bytes) bytes_pressure_ = true;
    }
    std::pair<uint64_t, uint64_t> targets(const HistoricalPressure &p) const {
        return {std::max(slots_pressure_ ? deficit(p.high_slots, p.ready_slots, p.pending_slots) : 0,
                         deficit(p.requested_ready_slots, p.ready_slots, p.pending_slots)),
                std::max(bytes_pressure_ ? deficit(p.high_cpu_bytes, p.free_cpu_bytes, p.pending_cpu_bytes) : 0,
                         deficit(p.scope_cpu_bytes, p.scope_cpu_limit, p.scope_pending_cpu_bytes))};
    }
    size_t capacity_, cursor_ = 0;
    Config config_;
    uint64_t incarnation_ = 0, round_ = 0;
    bool initialized_ = false, done_ = false, slots_pressure_ = false, bytes_pressure_ = false;
};

} // namespace vllm_kvmem
