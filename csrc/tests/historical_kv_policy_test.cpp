#include "historical_kv_policy.hpp"
#include "historical_policy_factory.hpp"

#include <atomic>
#include <array>
#include <chrono>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <thread>

using namespace vllm_kvmem;
using Repository = qw3::KvMemRepository;

static void check(bool value, const char *why) {
    if (!value) throw std::runtime_error(why);
}

static HistoricalBlockMeta block(uint32_t slot, uint64_t birth, uint64_t use) {
    HistoricalBlockMeta b;
    b.handle = {slot, 1};
    b.logical_id = slot + 1;
    b.birth_seq = birth;
    b.last_use_seq = use;
    b.cpu_bytes = 100;
    b.live = b.sealed = true;
    return b;
}

static void irreversible_retirement() {
    Repository repo({1, 32});
    auto gpu = repo.try_reserve(0, 32).commit();
    auto cpu = gpu.clone();
    const auto old = gpu.handle();
    check(repo.retire_permanently(old) == Repository::RetireResult::Committed,
          "nonzero references prevented permanent retirement");
    check(!repo.try_pin(old), "permanently deleted generation accepted a new reader");
    check(!repo.rescue(gpu), "old active lease resurrected permanent deletion");
    check(repo.retire_permanently(old) == Repository::RetireResult::AlreadyRetired,
          "repeated retirement was not idempotent");
    auto transfer = cpu.clone();
    check(transfer.record().token_begin == 0, "existing operation lost its payload identity");
    unsigned releases = 0;
    auto release = [&](const Repository::Record &) { ++releases; return true; };
    cpu.reset(); transfer.reset();
    check(repo.try_reclaim(old, release) == Repository::ReclaimResult::DeferredReaders && !releases,
          "metadata slot was reused during an active GPU operation");
    gpu.reset();
    check(!releases, "last foreground reference ran the release callback");
    std::thread worker([&] {
        check(repo.try_reclaim(old, release) == Repository::ReclaimResult::Reclaimed,
              "background reclaimer could not recycle deleted generation");
    });
    worker.join();
    check(releases == 1 && repo.ready_slots() == 1, "deleted slot was not returned exactly once");
    auto fresh = repo.try_reserve(32, 32).commit();
    check(repo.retire_permanently(old) == Repository::RetireResult::StaleHandle,
          "stale intent retired a reused slot");
    check(bool(repo.try_pin(fresh.handle())), "stale deletion hid fresh history");
    check(repo.retire_permanently({99, 1}) == Repository::RetireResult::StaleHandle,
          "out of range intent was accepted");
}

static void legacy_retirement_can_be_finalized() {
    Repository repo({1, 32});
    auto active = repo.try_reserve(0, 32).commit();
    check(repo.retire(active.handle()), "legacy retirement failed");
    check(repo.retire_permanently(active.handle()) == Repository::RetireResult::Committed,
          "legacy retirement could not be made irreversible");
    check(!repo.rescue(active), "legacy rescue reopened finalized history");
}

static void stale_intents_racing_with_slot_reuse_do_not_delete_new_generations() {
    constexpr uint32_t capacity = 4;
    constexpr uint64_t iterations = 2000;
    Repository repo({capacity, 32});
    std::array<std::atomic<uint64_t>, capacity> old_generations{};
    std::atomic<bool> stop{false}, corrupt{false};
    std::thread reclaimer([&] {
        while (!stop.load(std::memory_order_acquire)) {
            for (uint32_t slot = 0; slot < capacity; ++slot) {
                const auto h = repo.retired_at(slot);
                if (h.generation) repo.try_reclaim(h, [](const auto &) { return true; });
            }
            std::this_thread::yield();
        }
    });
    std::thread stale([&] {
        while (!stop.load(std::memory_order_acquire)) {
            for (uint32_t slot = 0; slot < capacity; ++slot) {
                const auto old = old_generations[slot].load(std::memory_order_acquire);
                if (old && repo.retire_permanently({slot, old}) == Repository::RetireResult::Committed)
                    corrupt.store(true);
            }
        }
    });
    struct Join {
        std::atomic<bool> &stop;
        std::thread &a, &b;
        ~Join() { stop.store(true, std::memory_order_release); a.join(); b.join(); }
    } join{stop, reclaimer, stale};
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    for (uint64_t i = 0; i < iterations; ++i) {
        Repository::Reservation reservation;
        while (!(reservation = repo.try_reserve(i*32, 32))) {
            check(std::chrono::steady_clock::now() < deadline, "stale intent test ran out of ready slots");
            std::this_thread::yield();
        }
        auto active = reservation.commit();
        Repository::RetireResult result;
        do { result = repo.retire_permanently(active.handle()); }
        while (result == Repository::RetireResult::Retry && std::chrono::steady_clock::now() < deadline);
        check(result == Repository::RetireResult::Committed, "fresh generation could not be retired");
        old_generations[active.handle().slot].store(active.handle().generation, std::memory_order_release);
    }
    while (repo.stats().reclaimed != iterations) {
        check(std::chrono::steady_clock::now() < deadline, "stale intent test did not drain");
        std::this_thread::yield();
    }
    check(!corrupt, "stale intent crossed a generation boundary during reuse");
}

static void policy_only_emits_intents() {
    HistoricalSnapshot s{7, 1, {block(0, 1, 4), block(1, 2, 2), block(2, 3, 3)}};
    s.blocks[1].gpu_resident = true;
    s.blocks[1].references = 8;
    HistoricalLruPolicy policy(8);
    HistoricalDecisionBatch out(2);
    HistoricalPressure pressure{0, 1, 2, 0};
    check(policy.step(s, pressure, {1, 1}, out).status == HistoricalPolicyStatus::Yielded,
          "policy failed to yield while scanning");
    policy.step(s, pressure, {1, 1}, out);
    check(policy.step(s, pressure, {1, 1}, out).emitted == 1, "policy lost oldest candidate");
    check(out.values()[0].handle.slot == 1, "GPU residency or references changed LRU order");
    check(s.blocks[1].live, "policy modified repository visibility");
    check(policy.step(s, pressure, {8, 8}, out).status == HistoricalPolicyStatus::Backpressure,
          "unacknowledged deletion batch was overwritten");
    out.clear_after_ack();
    pressure.pending_slots = 1;
    check(policy.step(s, pressure, {8, 8}, out).emitted == 1 &&
          out.values()[0].handle.slot == 2, "next batch did not retain candidate order");
    out.clear_after_ack();
    pressure.pending_slots = 2;
    check(policy.step(s, pressure, {8, 8}, out).status == HistoricalPolicyStatus::Complete,
          "pending slot credit did not stop excessive deletion");
}

static void byte_pressure_is_independent_of_slots() {
    HistoricalSnapshot s{7, 1, {block(0, 1, 1), block(1, 2, 2), block(2, 3, 3)}};
    HistoricalPressure pressure;
    pressure.ready_slots = 8;
    pressure.low_slots = 1;
    pressure.high_slots = 2;
    pressure.low_cpu_bytes = 100;
    pressure.high_cpu_bytes = 250;
    HistoricalLruPolicy policy(8);
    HistoricalDecisionBatch out(8);
    check(policy.step(s, pressure, {8, 8}, out).emitted == 3,
          "free metadata slots hid CPU byte pressure");
    out.clear_after_ack();
    ++s.round;
    pressure.pending_cpu_bytes = 250;
    check(policy.step(s, pressure, {8, 8}, out).emitted == 0,
          "pending CPU bytes were not deducted");
}

static void refill_reuses_one_scan_and_observes_current_pressure() {
    HistoricalSnapshot s{7, 1, {}};
    for (uint32_t i = 0; i < 1024; ++i) s.blocks.push_back(block(i, i, i));
    HistoricalPressure pressure{0, 128, 256, 0};
    HistoricalLruPolicy policy(1024);
    HistoricalDecisionBatch out(64);
    size_t examined = 0, emitted = 0;
    for (size_t tick = 0; tick < 100; ++tick) {
        auto result = policy.step(s, pressure, {256, 64}, out);
        examined += result.examined;
        for (const auto &intent : out.values()) {
            check(intent.handle.slot == emitted, "LRU batch order changed");
            ++emitted;
        }
        pressure.pending_slots += static_cast<uint32_t>(out.values().size());
        out.clear_after_ack();
        if (result.status == HistoricalPolicyStatus::Complete) break;
    }
    check(examined == 1024 && emitted == 256, "refill rescanned the repository per batch");

    HistoricalSnapshot small{9, 1, {block(0, 1, 1), block(1, 2, 2), block(2, 3, 3)}};
    HistoricalLruPolicy next(8);
    HistoricalDecisionBatch decisions(2);
    HistoricalPressure fresh{0, 1, 2, 0};
    next.step(small, fresh, {1, 2}, decisions);
    fresh.ready_slots = 2; // An unrelated request returned private reservations.
    check(next.step(small, fresh, {8, 2}, decisions).emitted == 0,
          "stale pressure caused deletion after capacity was returned");
}

static void eligibility_is_a_policy_choice_and_snapshot_is_stable() {
    HistoricalSnapshot s{7, 1, {block(0, 1, 1), block(1, 2, 2), block(2, 3, 3)}};
    s.blocks[0].mandatory = true;
    s.blocks[1].sealed = false;
    HistoricalPressure p{0, 1, 2, 0};
    HistoricalLruPolicy keep(8), allow(8, false);
    HistoricalDecisionBatch out(8);
    check(keep.step(s, p, {8, 8}, out).emitted == 1 && out.values()[0].handle.slot == 2,
          "default policy failed its explicit sealed/mandatory rules");
    out.clear_after_ack();
    check(allow.step(s, p, {8, 8}, out).emitted == 2 && out.values()[0].handle.slot == 0,
          "mandatory exclusion leaked into the retirement mechanism");
    out.clear_after_ack();
    HistoricalLruPolicy sliced(8);
    sliced.step(s, p, {1, 1}, out);
    ++s.round;
    bool rejected = false;
    try { sliced.step(s, p, {1, 1}, out); }
    catch (const std::logic_error &) { rejected = true; }
    check(rejected, "policy silently reset a partially scanned snapshot");
}

static void factory_validates_parameters_and_preserves_policy_behavior() {
    HistoricalSnapshot s{7, 1, {block(0, 1, 1), block(1, 2, 2)}};
    s.blocks[0].mandatory = true;
    const HistoricalPressure pressure{0, 1, 2, 0};
    HistoricalDecisionBatch out(8);
    auto keep = make_historical_kv_policy("lru", {}, 8);
    keep->step(s, pressure, {8, 8}, out);
    check(out.values().size() == 1 && out.values()[0].handle.slot == 1,
          "factory default changed mandatory protection");
    out.clear_after_ack();
    auto allow = make_historical_kv_policy("lru", {{"keep_mandatory", false}}, 8);
    allow->step(s, pressure, {8, 8}, out);
    check(out.values().size() == 2 && out.values()[0].handle.slot == 0,
          "factory ignored explicit LRU parameters");
    out.clear_after_ack();
    allow->reset();
    ++s.incarnation;
    ++s.round;
    allow->step(s, pressure, {8, 8}, out);
    check(out.values().size() == 2, "factory policy reset lost configuration or round state");
    out.clear_after_ack();
    auto legacy = make_historical_kv_policy("lru", {}, 8, false);
    legacy->step(s, pressure, {8, 8}, out);
    check(out.values().size() == 2, "legacy native keep_mandatory fallback was lost");
    out.clear_after_ack();
    auto explicit_keep = make_historical_kv_policy("lru", {{"keep_mandatory", true}}, 8, false);
    explicit_keep->step(s, pressure, {8, 8}, out);
    check(out.values().size() == 1, "explicit parameters did not override the legacy fallback");

    const std::vector<std::pair<std::string, HistoricalPolicyParameters>> invalid = {
        {"fifo", {}}, {"", {}}, {"lru", {{"typo", true}}},
        {"lru", {{"keep_mandatory", int64_t(1)}}},
        {"lru", {{"keep_mandatory", 1.0}}},
        {"lru", {{"keep_mandatory", std::string("true")}}}
    };
    for (const auto &[name, params] : invalid) {
        bool rejected = false;
        try { make_historical_kv_policy(name, params, 8); }
        catch (const std::invalid_argument &) { rejected = true; }
        check(rejected, "factory accepted an unknown policy or invalid parameter");
    }
}


static HistoricalEvidence evidence(uint64_t revision = 1) {
    return {revision, 5, 4, true, 0.05, -1.0};
}

static void statistical_requires_semantics_and_real_observations() {
    HistoricalSnapshot snapshot{1, 1, {}};
    for (uint32_t i = 0; i < 9; ++i) {
        auto b = block(i, i, i);
        b.evidence = evidence();
        snapshot.blocks.push_back(b);
    }
    snapshot.blocks[0].evidence = {}; // Unobserved is not cold.
    snapshot.blocks[1].evidence.eligible = false;
    snapshot.blocks[2].evidence.observations = 2;
    snapshot.blocks[2].evidence.cold_observations = 2;
    snapshot.blocks[3].evidence.cold_observations = 0; // A new important hit resets the run.
    snapshot.blocks[4].evidence.importance = 0.5;
    snapshot.blocks[5].mandatory = true;
    snapshot.blocks[6].sealed = false;
    snapshot.blocks[7].selection_holds = 1;
    HistoricalStatisticalPolicy policy(16);
    HistoricalPressure pressure;
    HistoricalDecisionBatch out(16);
    check(policy.wants_snapshot(pressure), "proactive statistical strategy required capacity pressure");
    const auto result = policy.step(snapshot, pressure, {16, 16}, out);
    check(result.examined == 9 && result.emitted == 1 && out.values()[0].handle.slot == 8,
          "statistical policy accepted absent, hot, unsealed or protected evidence");
    check(out.values()[0].evidence_revision == 1, "statistical intent lost evidence revision");
}

static void statistical_scans_and_decisions_are_bounded() {
    HistoricalSnapshot snapshot{1, 1, {}};
    for (uint32_t i = 0; i < 32; ++i) {
        auto b = block(i, i, i);
        b.evidence = evidence(i + 1);
        snapshot.blocks.push_back(b);
    }
    HistoricalStatisticalPolicy policy(32);
    HistoricalDecisionBatch out(3);
    HistoricalPressure pressure;
    size_t emitted = 0, examined = 0;
    for (unsigned round = 0; round < 32; ++round) {
        const auto result = policy.step(snapshot, pressure, {4, 2}, out);
        check(result.examined <= 4 && result.emitted <= 2, "statistical step exceeded a work budget");
        for (const auto &intent : out.values()) {
            check(intent.handle.slot == emitted && intent.evidence_revision == emitted + 1,
                  "statistical scan lost or repeated a candidate");
            ++emitted;
        }
        examined += result.examined;
        if (!out.empty())
            check(policy.step(snapshot, pressure, {4, 2}, out).status == HistoricalPolicyStatus::Backpressure,
                  "unacknowledged statistical decisions were overwritten");
        out.clear_after_ack();
        if (result.status != HistoricalPolicyStatus::Yielded) break;
    }
    check(examined == 32 && emitted == 32, "bounded statistical round did not finish");
}

static void statistical_pressure_and_optional_model_gate_are_conservative() {
    HistoricalSnapshot snapshot{1, 1, {block(0, 0, 0), block(1, 1, 1), block(2, 2, 2)}};
    for (auto &b : snapshot.blocks) b.evidence = evidence();
    HistoricalStatisticalPolicy::Config config;
    config.proactive = false;
    HistoricalStatisticalPolicy policy(8, config);
    HistoricalDecisionBatch out(8);
    HistoricalPressure pressure;
    check(!policy.wants_snapshot(pressure), "pressure-only strategy ran without pressure");
    check(policy.step(snapshot, pressure, {8, 8}, out).emitted == 0, "pressure-only strategy proactively deleted");
    pressure = {0, 1, 2, 0};
    ++snapshot.round;
    check(policy.wants_snapshot(pressure), "statistical strategy ignored pressure");
    check(policy.step(snapshot, pressure, {8, 8}, out).emitted == 2,
          "pressure-only strategy failed to stop at capacity target");
    out.clear_after_ack();
    ++snapshot.round;
    pressure.pending_slots = 2;
    check(policy.step(snapshot, pressure, {8, 8}, out).emitted == 0,
          "pending retirement was counted as an additional deletion deficit");
    config.proactive = true;
    config.min_model_reclaim_score = 0.9;
    HistoricalStatisticalPolicy model_policy(8, config);
    snapshot.blocks[1].evidence.model_reclaim_score = 0.5;
    snapshot.blocks[2].evidence.model_reclaim_score = 0.95;
    check(model_policy.step(snapshot, {}, {8, 8}, out).emitted == 1 &&
          out.values()[0].handle.slot == 2, "unknown model score was treated as a confident reclaim signal");
}

static void statistical_factory_rejects_invalid_or_unsafe_parameters() {
    const std::vector<HistoricalPolicyParameters> invalid = {
        {{"keep_mandatory", false}}, {{"min_observations", true}},
        {{"min_observations", 3.0}}, {{"min_observations", int64_t(0)}},
        {{"min_cold_observations", int64_t(-1)}}, {{"proactive", int64_t(1)}},
        {{"max_importance", true}}, {{"max_importance", std::numeric_limits<double>::infinity()}},
        {{"max_importance", std::numeric_limits<double>::quiet_NaN()}},
        {{"max_importance", -0.1}}, {{"max_importance", 1.1}},
        {{"min_model_reclaim_score", -0.2}}, {{"min_model_reclaim_score", 1.1}},
        {{"typo", true}}
    };
    for (const auto &params : invalid) {
        bool rejected = false;
        try { make_historical_kv_policy("statistical", params, 8); }
        catch (const std::invalid_argument &) { rejected = true; }
        check(rejected, "statistical factory accepted an invalid parameter");
    }
    auto policy = make_historical_kv_policy("statistical", {
        {"min_observations", int64_t(1)}, {"min_cold_observations", int64_t(1)},
        {"max_importance", 1.0}, {"proactive", true}, {"min_model_reclaim_score", -1.0}}, 8, false);
    auto b = block(0, 0, 0);
    b.evidence = evidence();
    b.mandatory = true;
    check(!policy->permits(b), "legacy keep_mandatory=false weakened the statistical policy");
}


static void explicit_artifact_profile_does_not_fabricate_cold_observations() {
    auto policy = make_historical_kv_policy("statistical", {
        {"min_observations", int64_t(1)}, {"min_cold_observations", int64_t(0)},
        {"max_importance", 1.0}}, 8);
    auto b = block(0, 0, 0);
    b.evidence = {1, 1, 0, true, 1.0, -1.0};
    check(policy->permits(b), "explicit artifact profile still required a fake cold sample");
    b.evidence.observations = 0;
    check(!policy->permits(b), "artifact profile accepted an unobserved block");
    b.evidence.observations = 1;
    b.evidence.eligible = false;
    check(!policy->permits(b), "artifact profile bypassed semantic eligibility");
    b.evidence.eligible = true;
    b.selection_holds = 1;
    check(!policy->permits(b), "artifact profile bypassed protected result/selection");
    b.selection_holds = 0;
    HistoricalStatisticalPolicy defaults(8);
    check(!defaults.permits(b), "artifact experiment weakened default cold thresholds");
}

int main() {
    explicit_artifact_profile_does_not_fabricate_cold_observations();
    try {
        statistical_requires_semantics_and_real_observations();
        statistical_scans_and_decisions_are_bounded();
        statistical_pressure_and_optional_model_gate_are_conservative();
        statistical_factory_rejects_invalid_or_unsafe_parameters();
        irreversible_retirement();
        legacy_retirement_can_be_finalized();
        stale_intents_racing_with_slot_reuse_do_not_delete_new_generations();
        policy_only_emits_intents();
        byte_pressure_is_independent_of_slots();
        refill_reuses_one_scan_and_observes_current_pressure();
        eligibility_is_a_policy_choice_and_snapshot_is_stable();
        factory_validates_parameters_and_preserves_policy_behavior();
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    std::cout << "Historical KV irreversible retirement and LRU policy tests passed\n";
}
