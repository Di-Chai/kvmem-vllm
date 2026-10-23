#include "historical_manager.hpp"

#include <array>
#include <atomic>
#include <functional>
#include <limits>
#include <chrono>
#include <iostream>
#include <stdexcept>
#include <thread>

using namespace vllm_kvmem;
using Repo = qw3::KvMemRepository;
using Payload = HistoricalPayloadArena;

static void check(bool value, const char *why) {
    if (!value) throw std::runtime_error(why);
}

template <typename Predicate>
static void eventually(Predicate predicate, const char *why) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (!predicate()) {
        check(std::chrono::steady_clock::now() < deadline, why);
        std::this_thread::yield();
    }
}

struct Fixture {
    std::shared_ptr<Payload> arena;
    std::shared_ptr<Repo> repo;
    std::unique_ptr<HistoricalGcWorker> worker;
    std::shared_ptr<HistoricalGcScope> scope;
    std::vector<Repo::Lease> gpu;
    std::vector<qw3::KvMemBlockHandle> handles;
    uint64_t payload_epoch = 0;

    Fixture(HistoricalGcMode mode = HistoricalGcMode::Delete, bool background = false,
            std::unique_ptr<HistoricalKvPolicy> policy = {}, uint32_t max_scopes = 1, bool preallocate = false,
            HistoricalPolicyParameters params = {}, std::string policy_name = "lru") {
        arena = std::make_shared<Payload>(Payload::Config{16, 16, 1, 16, 16, background});
        repo = std::make_shared<Repo>(Repo::Config{8, 4});
        HistoricalGcWorker::Config config;
        config.max_scopes = max_scopes;
        config.scope_capacity = preallocate ? 8 : 0;
        config.poll_ms = 1;
        config.ready_low = 2;
        config.ready_high = 4;
        config.scan_blocks = 8;
        config.delete_blocks = 2;
        config.mode = mode;
        config.policy_params = std::move(params);
        config.policy = std::move(policy_name);
        config.background = background;
        worker = std::make_unique<HistoricalGcWorker>(config, arena);
        scope = worker->register_scope(repo, std::move(policy));
        check(bool(scope), "scope registration failed");
    }
    void append(uint32_t flags = HistoricalSealed | HistoricalGpu, uint64_t use = 0) {
        const auto index = handles.size();
        auto lease = repo->try_reserve(index * 4, 4).commit();
        const Payload::Handle backing{static_cast<uint32_t>(index), payload_epoch + index + 1};
        std::array<uint8_t, 16> bytes{};
        bytes.fill(uint8_t(index));
        arena->publish(backing, bytes.data(), bytes.size());
        auto cpu = arena->acquire(backing);
        check(scope->publish(lease, cpu, flags, use), "backing publication failed");
        handles.push_back(lease.handle());
        gpu.push_back(std::move(lease));
    }
    void step(unsigned times = 1, bool ack = true) {
        for (unsigned i = 0; i < times; ++i) {
            worker->advance();
            arena->advance();
            if (ack) for (const auto &event : scope->completions(8)) scope->acknowledge(event);
        }
    }
    ~Fixture() {
        gpu.clear();
        if (scope) eventually([&] { worker->advance(); return scope->close(); }, "scope cleanup stalled");
        scope.reset();
        if (worker) eventually([&] { return worker->close(); }, "worker cleanup stalled");
        worker.reset();
        eventually([&] { return arena->close(); }, "arena cleanup stalled");
    }
};

static void lru_retirement_and_cpu_reclamation_do_not_wait_for_gpu() {
    Fixture fixture;
    for (unsigned i = 0; i < 8; ++i) fixture.append();
    fixture.step(12);
    const auto stats = fixture.scope->stats();
    check(stats.retired == 4 && stats.cpu_released == 4, "LRU did not retire/release exactly the shortage");
    check(stats.pending_slots == 4 && stats.slots_released == 0, "active GPU metadata was recycled");
    check(fixture.arena->stats().used_bytes == 64, "CPU bytes remained tied to active GPU leases");
    for (unsigned i = 0; i < 4; ++i) {
        check(!fixture.repo->try_pin(fixture.handles[i]), "LRU order was not oldest first");
        check(fixture.gpu[i].record().token_begin == i * 4, "active GPU operation lost its identity");
    }
    for (unsigned i = 4; i < 8; ++i) check(bool(fixture.repo->try_pin(fixture.handles[i])), "LRU deleted beyond target");
    fixture.gpu.clear();
    fixture.step(12);
    check(fixture.repo->ready_slots() == 4 && fixture.scope->stats().pending_slots == 0,
          "last GPU completion did not return metadata slots");
}

static void completions_are_reliable_until_acknowledged_and_across_reuse() {
    Fixture fixture;
    for (unsigned i = 0; i < 8; ++i) fixture.append();
    fixture.step(12, false);
    fixture.gpu.clear();
    fixture.step(4, false);
    check(fixture.repo->ready_slots() == 0, "repository reused a slot before CPU cleanup acknowledgement");
    auto cpu = fixture.scope->completions(8);
    check(cpu.size() == 4, "unconsumed CPU completion was lost");
    for (const auto &event : cpu) check(event.cpu && !event.slot, "unexpected completion phase");
    fixture.step(2, false);
    check(fixture.repo->ready_slots() == 0, "merely observing completion acknowledged cleanup");
    for (const auto &event : cpu) check(fixture.scope->acknowledge(event), "CPU completion acknowledgement failed");
    fixture.step(2, false);
    check(fixture.repo->ready_slots() == 4, "acknowledged CPU cleanup did not permit repository reuse");
    auto replacement = fixture.repo->try_reserve(32, 4).commit();
    const auto new_handle = replacement.handle();
    const auto old = fixture.handles[new_handle.slot];
    std::array<uint8_t, 16> bytes{};
    fixture.arena->publish({8, 9}, bytes.data(), bytes.size());
    auto payload = fixture.arena->acquire({8, 9});
    check(fixture.scope->publish(replacement, payload, HistoricalSealed | HistoricalGpu),
          "new generation could not publish before old slot acknowledgement");
    payload.reset();
    auto slots = fixture.scope->completions(8);
    bool found = false;
    for (const auto &event : slots) {
        if (event.handle == old && event.slot) found = true;
        fixture.scope->acknowledge(event);
    }
    check(found, "new generation overwrote an unconsumed old slot completion");
    check(bool(fixture.repo->try_pin(new_handle)), "old completion deleted replacement history");
    replacement.reset();
}

static void off_and_shadow_do_not_change_real_history() {
    for (const auto mode : {HistoricalGcMode::Off, HistoricalGcMode::Shadow}) {
        Fixture fixture(mode);
        for (unsigned i = 0; i < 8; ++i) fixture.append();
        fixture.step(32);
        for (const auto handle : fixture.handles) check(bool(fixture.repo->try_pin(handle)), "off/shadow hid real KV");
        check(fixture.arena->stats().used_bytes == 128, "off/shadow freed real CPU bytes");
        const auto stats = fixture.scope->stats();
        check(stats.retired == 0 && stats.cpu_released == 0, "off/shadow reported real retirement");
        check(mode == HistoricalGcMode::Off ? stats.decisions == 0 : stats.shadow_decisions == 4,
              "off/shadow policy accounting was incorrect");
    }
}

class DeletePublished final : public HistoricalKvPolicy {
public:
    HistoricalPolicyResult step(const HistoricalSnapshot &snapshot, const HistoricalPressure &,
                                HistoricalWorkBudget, HistoricalDecisionBatch &out) override {
        if (snapshot.blocks.empty()) return {HistoricalPolicyStatus::NoCandidate, 0, 0};
        out.push({snapshot.blocks.front().handle, snapshot.incarnation, snapshot.round});
        return {HistoricalPolicyStatus::Complete, 1, 1};
    }
};

static void proactive_policy_can_delete_unsealed_active_blocks_without_pressure() {
    Fixture fixture(HistoricalGcMode::Delete, false, std::make_unique<DeletePublished>());
    fixture.append(HistoricalGpu | HistoricalMandatory);
    fixture.step(8);
    check(fixture.scope->stats().retired == 1 && fixture.scope->stats().cpu_released == 1,
          "manager vetoed a custom strategy by watermarks, active, sealed or mandatory flags");
    check(fixture.gpu[0].record().token_begin == 0, "proactive deletion interrupted old operation");
}

class ThrowAfterOne final : public HistoricalKvPolicy {
    bool emitted_ = false;
public:
    HistoricalPolicyResult step(const HistoricalSnapshot &snapshot, const HistoricalPressure &,
                                HistoricalWorkBudget, HistoricalDecisionBatch &out) override {
        if (emitted_) throw std::runtime_error("injected policy failure");
        emitted_ = true;
        out.push({snapshot.blocks.front().handle, snapshot.incarnation, snapshot.round});
        return {HistoricalPolicyStatus::Complete, 1, 1};
    }
};

static void policy_failure_and_pause_do_not_stop_committed_releases() {
    Fixture fixture(HistoricalGcMode::Delete, false, std::make_unique<ThrowAfterOne>());
    fixture.append();
    fixture.step(8);
    check(fixture.scope->stats().policy_failures == 1, "policy exception was not surfaced");
    check(fixture.scope->stats().cpu_released == 1, "policy failure prevented CPU reclamation");
    fixture.scope->pause_policy(true);
    fixture.gpu.clear();
    fixture.step(8);
    check(fixture.scope->stats().slots_released == 1, "paused strategy prevented old metadata release");
}

static void original_io_can_publish_after_retirement_without_resurrection() {
    Fixture fixture(HistoricalGcMode::Off);
    auto io = fixture.repo->try_reserve(0, 4).commit();
    const auto handle = io.handle();
    check(fixture.repo->retire_permanently(handle) == Repo::RetireResult::Committed, "retirement failed");
    fixture.step(2);
    std::array<uint8_t, 16> bytes{};
    fixture.arena->publish({0, 1}, bytes.data(), bytes.size());
    auto cpu = fixture.arena->acquire({0, 1});
    check(fixture.scope->publish(io, cpu, HistoricalSealed), "late I/O could not register its real ownership");
    cpu.reset();
    fixture.step(8);
    check(!fixture.repo->try_pin(handle) && fixture.arena->stats().used_bytes == 0,
          "late publication reopened retired history or leaked CPU ownership");
    check(fixture.repo->ready_slots() == 7, "in-flight original I/O was freed early");
    io.reset();
    fixture.step(8);
    check(fixture.repo->ready_slots() == 8, "late I/O completion did not release its metadata");
}

static void foreign_owners_and_overwrites_are_rejected_and_scope_slots_recycle() {
    Fixture fixture(HistoricalGcMode::Off);
    fixture.append();
    auto other = std::make_shared<Repo>(Repo::Config{8, 4});
    auto foreign = other->try_reserve(0, 4).commit();
    auto cpu = fixture.arena->acquire({0, 1});
    bool rejected = false;
    try { fixture.scope->publish(foreign, cpu, HistoricalSealed); }
    catch (const std::invalid_argument &) { rejected = true; }
    check(rejected, "foreign repository with identical handle was accepted");
    check(fixture.scope->publish(fixture.gpu[0], cpu, HistoricalSealed), "idempotent backing registration failed");
    check(fixture.scope->stats().published == 1, "repeated publication double-counted backing");
    cpu.reset();
    fixture.gpu.clear();
    eventually([&] { fixture.worker->advance(); return fixture.scope->close(); }, "scope close stalled");
    fixture.scope.reset();
    fixture.worker->advance();
    fixture.scope = fixture.worker->register_scope(fixture.repo);
    check(bool(fixture.scope), "fixed scope registry did not recycle a completed namespace");
}

static void background_manager_progresses_without_foreground_policy_calls() {
    Fixture fixture(HistoricalGcMode::Delete, true);
    for (unsigned i = 0; i < 8; ++i) fixture.append();
    eventually([&] { return fixture.scope->stats().cpu_released == 4; }, "native background manager made no CPU progress");
    check(fixture.repo->ready_slots() == 0, "background manager released active GPU metadata");
    fixture.gpu.clear();
    eventually([&] {
        for (const auto &event : fixture.scope->completions(8)) fixture.scope->acknowledge(event);
        return fixture.scope->stats().slots_released == 4;
    },
               "native background manager made no metadata progress");
}

static void store_can_finish_and_stage_out_a_retired_gpu_window() {
    Fixture fixture;
    qw3::KvMemStoreConfig config;
    config.block_tokens = 4;
    config.select_budget = 16;
    config.gen_budget = 16;
    config.sink_blocks = 1;
    config.recent_blocks = 1;
    config.optimize_stage_in = true;
    auto store = std::make_unique<qw3::KvMemStore>(config, fixture.repo);
    store->attach_history_observer(fixture.scope);
    auto append = store->prepare_repository_append(32, 0);
    check(bool(append), "store append reservation failed");
    append->commit(32);
    append.reset();
    for (unsigned i = 0; i < 8; ++i) {
        auto io = fixture.repo->try_pin_slot(i);
        std::array<uint8_t, 16> bytes{};
        fixture.arena->publish({i, i + 1}, bytes.data(), bytes.size());
        auto cpu = fixture.arena->acquire({i, i + 1});
        check(fixture.scope->publish(io, cpu, 0), "I/O backing publication failed");
        fixture.handles.push_back(io.handle());
    }
    store->gc_begin_step();
    fixture.step(16);
    check(fixture.scope->stats().cpu_released == 4 && fixture.scope->stats().slots_released == 0,
          "CPU reclamation interfered with Store GPU execution rights");
    check(bool(fixture.repo->try_pin(fixture.handles[0])) && bool(fixture.repo->try_pin(fixture.handles[7])),
          "LRU ignored Store sink/recent policy metadata");
    bool rejected = false;
    try { store->set_repository_selection({fixture.handles[1]}); }
    catch (const std::runtime_error &) { rejected = true; }
    check(rejected, "old GPU ownership granted a retired block new selection rights");
    auto plan = store->set_repository_selection({fixture.handles[0], fixture.handles[5],
                                                fixture.handles[6], fixture.handles[7]});
    check(plan.stage_out.size() == 4 && plan.total_window_tokens == 16,
          "normal refresh failed to include retired old GPU pages in stage-out");
    for (auto slot : plan.stage_out)
        store->set_block_tier(slot, qw3::KvTier::SSD, -1, -1, fixture.handles[slot].generation);
    fixture.step(4);
    check(fixture.scope->stats().slots_released == 0, "previous-window fence was bypassed");
    plan.repository_previous_window.reset();
    fixture.step(8);
    check(fixture.scope->stats().slots_released == 4, "ordinary refresh did not release retired GPU generations");
    store.reset();
}

static void engine_metadata_does_not_contend_with_io_publication() {
    Fixture fixture(HistoricalGcMode::Off);
    fixture.append();
    auto cpu = fixture.arena->acquire({0, 1});
    std::thread io([&] {
        for (unsigned i = 0; i < 10000; ++i)
            check(fixture.scope->publish(fixture.gpu[0], cpu, HistoricalSealed), "sole I/O publisher contended");
    });
    for (unsigned i = 0; i < 10000; ++i) {
        fixture.scope->used(i);
        check(fixture.scope->update(fixture.gpu[0], HistoricalSealed | HistoricalGpu),
              "engine metadata acquired the I/O publication guard");
    }
    io.join();
    cpu.reset();
}

static void execution_admission_races_retirement_and_generation_reuse() {
    Fixture fixture(HistoricalGcMode::Off);
    for (uint64_t generation = 1; generation <= 500; ++generation) {
        auto gpu = fixture.repo->try_reserve(generation * 4, 4).commit();
        const auto handle = gpu.handle();
        std::array<uint8_t, 16> bytes{};
        bytes.fill(uint8_t(generation));
        fixture.arena->publish({0, generation}, bytes.data(), bytes.size());
        auto cpu = fixture.arena->acquire({0, generation});
        check(fixture.scope->publish(gpu, cpu, HistoricalSealed | HistoricalGpu), "race publication failed");
        cpu.reset();
        std::thread deletion([&] {
            while (fixture.repo->retire_permanently(handle) == Repo::RetireResult::Retry) std::this_thread::yield();
        });
        auto admission = fixture.scope->admit(handle);
        deletion.join();
        fixture.step(4);
        if (admission.lease) {
            auto source = std::dynamic_pointer_cast<HistoricalReadOwner>(admission.cpu);
            check(bool(source) && source->read.data()[0] == uint8_t(generation), "admission lost its actual CPU owner");
            check(fixture.arena->stats().used_bytes == 16, "admitted operation's CPU bytes were freed");
        }
        check(!fixture.scope->admit(handle).lease, "retired generation obtained a new execution admission");
        admission = {};
        gpu.reset();
        fixture.step(8);
        check(fixture.repo->ready_slots() == 8 && fixture.arena->stats().used_bytes == 0,
              "admission race leaked CPU or repository capacity");
    }
}

static void policy_frame_holds_revalidate_old_decisions_and_release_without_cpu_pins() {
    Fixture fixture;
    for (unsigned i = 0; i < 8; ++i) fixture.append();
    fixture.step(); // The immutable heap already contains all eight candidates.
    auto frame = fixture.scope->protect({fixture.handles[2], fixture.handles[3]});
    check(bool(frame), "checkpoint policy admission failed");
    fixture.step(24);
    check(fixture.scope->stats().retired == 4, "revalidation did not find replacement LRU candidates");
    for (auto i : {2, 3, 6, 7}) check(bool(fixture.repo->try_pin(fixture.handles[i])), "LRU deleted a protected frame");
    for (auto i : {0, 1, 4, 5}) check(!fixture.repo->try_pin(fixture.handles[i]), "LRU order ignored delayed eligibility");
    check(fixture.arena->stats().readers == 0, "policy retention acquired CPU read rights");
    frame.reset();
}

static void policy_frame_hold_is_optional_for_the_retirement_mechanism() {
    Fixture fixture(HistoricalGcMode::Delete, false, std::make_unique<DeletePublished>());
    fixture.append();
    auto frame = fixture.scope->protect({fixture.handles[0]});
    check(bool(frame), "custom-policy frame admission failed");
    fixture.step(8);
    check(fixture.scope->stats().cpu_released == 1, "mechanism vetoed a custom policy by checkpoint ownership");
    check(!fixture.scope->protect({fixture.handles[0]}), "retired checkpoint obtained new protection rights");
    frame.reset();
}

static void failed_frame_admission_rolls_back_earlier_policy_holds() {
    Fixture fixture;
    for (unsigned i = 0; i < 8; ++i) fixture.append();
    auto invalid = fixture.handles[1];
    ++invalid.generation;
    check(!fixture.scope->protect({fixture.handles[0], invalid}), "stale frame was admitted");
    fixture.step(16);
    check(!fixture.repo->try_pin(fixture.handles[0]), "partial failed admission leaked a policy hold");
}

static void requested_capacity_can_exceed_static_lru_watermarks() {
    Fixture fixture;
    for (unsigned i = 0; i < 8; ++i) fixture.append();
    fixture.scope->request_capacity(6);
    fixture.step(24);
    check(fixture.scope->stats().retired == 6 && fixture.scope->stats().cpu_released == 6,
          "waiting turn could not request a reserve larger than the static high watermark");
    check(fixture.repo->ready_slots() == 0, "capacity demand bypassed the old GPU fence");
    fixture.gpu.clear();
    fixture.step(8);
    check(fixture.repo->ready_slots() == 6, "fenced capacity demand did not become ready supply");
    fixture.scope->request_capacity(0);
    fixture.step(8);
    check(fixture.scope->stats().retired == 6, "cleared admission demand continued deleting history");
}

static void scope_cpu_target_is_independent_of_global_free_bytes() {
    Fixture fixture;
    for (unsigned i = 0; i < 8; ++i) fixture.append();
    auto frame = fixture.scope->protect({fixture.handles[0]});
    fixture.scope->request_capacity(0, 32);
    fixture.step(24);
    check(fixture.arena->stats().used_bytes == 32 && fixture.scope->stats().cpu_released == 6,
          "per-scope CPU demand did not stop at its requested retained size");
    check(bool(fixture.repo->try_pin(fixture.handles[0])), "CPU reservation deleted an LRU mandatory checkpoint");
    fixture.scope->request_capacity(0);
    fixture.step(8);
    check(fixture.scope->stats().retired == 6, "reset CPU demand was not cleared");
    frame.reset();
}

static void serving_metadata_is_preallocated_and_recycled_before_scope_ack(bool keep_mandatory = true) {
    Fixture fixture(HistoricalGcMode::Delete, false, {}, 1, true, {{"keep_mandatory", keep_mandatory}});
    check(fixture.worker->storage_stats().allocations == 1, "serving metadata was not allocated at startup");
    for (unsigned round = 0; round < 10; ++round) {
        fixture.payload_epoch = uint64_t(round)*8;
        for (unsigned i = 0; i < 8; ++i)
            fixture.append(HistoricalSealed | HistoricalGpu | (keep_mandatory ? 0 : HistoricalMandatory));
        fixture.step(16);
        check(fixture.scope->stats().retired == 4, "reused LRU policy inherited its previous incarnation");
        fixture.gpu.clear();
        eventually([&] { fixture.worker->advance(); return fixture.scope->close(); }, "pooled metadata close did not finish");
        const auto storage = fixture.worker->storage_stats();
        check(storage.allocations == 1 && storage.recycles == round+1,
              "scope registration allocated large metadata or its ACK preceded recycling");
        fixture.scope.reset();
        fixture.handles.clear();
        fixture.repo = std::make_shared<Repo>(Repo::Config{8, 4});
        fixture.scope = fixture.worker->register_scope(fixture.repo);
        check(bool(fixture.scope), "acknowledged pooled scope was not immediately reusable");
    }
}


static HistoricalEvidence cold_evidence(uint64_t revision = 1) {
    return {revision, 5, 4, true, 0.05, -1.0};
}

static void selection_holds_are_a_mechanism_veto() {
    for (bool custom : {false, true}) {
        Fixture fixture(HistoricalGcMode::Delete, false,
                        custom ? std::make_unique<DeletePublished>() : nullptr,
                        1, false, {{"keep_mandatory", false}});
        fixture.append(HistoricalSealed | HistoricalMandatory);
        fixture.scope->request_capacity(8);
        auto hold = fixture.scope->protect_selection({fixture.handles[0]});
        check(bool(hold), "selection transaction could not be admitted");
        fixture.step(16);
        check(fixture.scope->stats().retired == 0 && bool(fixture.repo->try_pin(fixture.handles[0])),
              "custom policy or keep_mandatory=false bypassed a selection hold");
        check(fixture.arena->stats().readers == 0, "selection hold acquired CPU payload ownership");
        hold.reset();
        fixture.step(16);
        check(fixture.scope->stats().retired == 1, "released selection hold permanently suppressed deletion");
    }
}

static void selection_failure_close_and_scope_reuse_are_safe() {
    Fixture fixture(HistoricalGcMode::Off, false, {}, 1, true);
    fixture.append();
    auto bad = fixture.handles[0];
    ++bad.generation;
    check(!fixture.scope->protect_selection({fixture.handles[0], bad}),
          "partial selection accepted a stale generation");
    auto hold = fixture.scope->protect_selection({fixture.handles[0]});
    check(bool(hold), "valid selection admission failed after rollback");
    check(!fixture.scope->close(), "scope closed with an active selection");
    fixture.step(8);
    check(bool(fixture.repo->try_pin(fixture.handles[0])), "close retired an admitted selection");
    check(!fixture.scope->protect_selection({fixture.handles[0]}), "closing scope admitted a new selection");
    check(!fixture.scope->publish_evidence(fixture.handles[0], cold_evidence()),
          "closing scope accepted new evidence");
    hold.reset();
    fixture.gpu.clear();
    eventually([&] { fixture.worker->advance(); return fixture.scope->close(); }, "selection rollback leaked a close hold");
    fixture.scope.reset();
    check(fixture.worker->storage_stats().recycles == 1, "old selection state escaped scope recycling");
    fixture.repo = std::make_shared<Repo>(Repo::Config{8, 4});
    fixture.scope = fixture.worker->register_scope(fixture.repo);
    fixture.handles.clear();
    fixture.payload_epoch = 8;
    fixture.append();
    auto fresh = fixture.scope->protect_selection({fixture.handles[0]});
    check(bool(fresh), "reused scope retained a stale selection count");
    fresh.reset();
}

static void selection_admission_races_native_retirement() {
    for (unsigned attempt = 0; attempt < 64; ++attempt) {
        Fixture fixture(HistoricalGcMode::Delete, false, std::make_unique<DeletePublished>());
        fixture.append();
        std::atomic<bool> go{false};
        std::thread worker([&] {
            while (!go.load(std::memory_order_acquire)) std::this_thread::yield();
            for (unsigned i = 0; i < 8; ++i) fixture.worker->advance();
        });
        go.store(true, std::memory_order_release);
        auto hold = fixture.scope->protect_selection({fixture.handles[0]});
        worker.join();
        if (hold) check(bool(fixture.repo->try_pin(fixture.handles[0])),
                        "successful selection admission raced past permanent retirement");
        hold.reset();
        fixture.step(12);
        check(!fixture.scope->protect_selection({fixture.handles[0]}),
              "retired generation accepted a selection hold");
    }
}

static void statistical_evidence_is_required_and_revision_qualified() {
    Fixture fixture(HistoricalGcMode::Delete, false, {}, 1, false, {}, "statistical");
    fixture.append();
    fixture.step(64);
    check(fixture.scope->stats().retired == 0, "unobserved history aged into statistical eligibility");
    auto e = cold_evidence();
    e.observations = e.cold_observations = 0;
    check(fixture.scope->publish_evidence(fixture.handles[0], e), "initial evidence was rejected");
    fixture.step(64);
    check(fixture.scope->stats().retired == 0, "GC polls manufactured observations");
    e = cold_evidence(2);
    e.eligible = false;
    check(fixture.scope->publish_evidence(fixture.handles[0], e), "new protected evidence was rejected");
    check(!fixture.scope->publish_evidence(fixture.handles[0], cold_evidence(1)),
          "older evidence overwrote a new protected revision");
    check(!fixture.scope->publish_evidence(fixture.handles[0], cold_evidence(2)),
          "duplicate revision changed evidence");
    fixture.step(16);
    check(fixture.scope->stats().retired == 0, "semantic eligibility was ignored");
    check(fixture.scope->publish_evidence(fixture.handles[0], cold_evidence(3)), "eligible evidence was rejected");
    fixture.step(16);
    check(fixture.scope->stats().retired == 1, "proactive statistical evidence did not retire a block");
    check(!fixture.scope->publish_evidence(fixture.handles[0], cold_evidence(4)),
          "late evidence resurrected retired history");
}

class UpdateEvidenceAfterDecision final : public HistoricalKvPolicy {
public:
    HistoricalStatisticalPolicy inner{8};
    std::function<void()> update;
    bool permits(const HistoricalBlockMeta &meta) const override { return inner.permits(meta); }
    HistoricalPolicyResult step(const HistoricalSnapshot &snapshot, const HistoricalPressure &pressure,
                                HistoricalWorkBudget budget, HistoricalDecisionBatch &out) override {
        auto result = inner.step(snapshot, pressure, budget, out);
        if (!out.empty() && update) { auto callback = std::move(update); callback(); }
        return result;
    }
};

static void changed_evidence_vetoes_a_delayed_statistical_intent() {
    auto policy = std::make_unique<UpdateEvidenceAfterDecision>();
    auto *raw = policy.get();
    Fixture fixture(HistoricalGcMode::Delete, false, std::move(policy));
    fixture.append();
    check(fixture.scope->publish_evidence(fixture.handles[0], cold_evidence()), "test evidence publication failed");
    raw->update = [&] {
        auto protected_evidence = cold_evidence(2);
        protected_evidence.eligible = false;
        check(fixture.scope->publish_evidence(fixture.handles[0], protected_evidence),
              "decision-time evidence update failed");
    };
    fixture.step(16);
    check(fixture.scope->stats().decisions == 1 && fixture.scope->stats().retired == 0,
          "old snapshot intent deleted newly protected history");
    check(fixture.scope->publish_evidence(fixture.handles[0], cold_evidence(3)), "replacement evidence failed");
    fixture.step(16);
    check(fixture.scope->stats().retired == 1, "fresh evidence did not recover after a stale decision");
}

static void evidence_revision_changes_force_a_fresh_decision() {
    auto policy = std::make_unique<UpdateEvidenceAfterDecision>();
    auto *raw = policy.get();
    Fixture fixture(HistoricalGcMode::Delete, false, std::move(policy));
    fixture.append();
    check(fixture.scope->publish_evidence(fixture.handles[0], cold_evidence()), "initial evidence failed");
    raw->update = [&] {
        check(fixture.scope->publish_evidence(fixture.handles[0], cold_evidence(2)),
              "equally eligible newer evidence failed");
    };
    fixture.step(1);
    check(fixture.scope->stats().decisions == 1 && fixture.scope->stats().retired == 0,
          "revision-qualified intent bypassed a concurrent evidence update");
    fixture.step(16);
    check(fixture.scope->stats().retired == 1, "fresh round failed after revision invalidation");
}

static void statistical_evidence_does_not_cross_generation_reuse() {
    Fixture fixture(HistoricalGcMode::Delete, false, {}, 1, false, {}, "statistical");
    fixture.append();
    const auto old = fixture.handles[0];
    check(fixture.scope->publish_evidence(old, cold_evidence(9)), "old generation evidence failed");
    fixture.gpu.clear();
    fixture.step(16);
    // Fill all available slots to force reuse of the old slot.
    std::vector<Repo::Lease> replacements;
    Repo::Lease replacement;
    for (unsigned i = 0; i < 8; ++i) {
        auto value = fixture.repo->try_reserve(4 + i * 4, 4).commit();
        check(bool(value), "replacement reservation failed");
        if (value.handle().slot == old.slot) replacement = std::move(value);
        else replacements.push_back(std::move(value));
    }
    check(bool(replacement) && replacement.handle().generation != old.generation, "test did not recycle old metadata");
    std::array<uint8_t, 16> bytes{};
    fixture.arena->publish({0, 20}, bytes.data(), bytes.size());
    auto cpu = fixture.arena->acquire({0, 20});
    check(fixture.scope->publish(replacement, cpu, HistoricalSealed), "replacement CPU publication failed");
    cpu.reset();
    check(!fixture.scope->publish_evidence(old, cold_evidence(10)), "old generation evidence targeted a new slot");
    fixture.step(16);
    check(bool(fixture.repo->try_pin(replacement.handle())), "reused slot inherited old statistical eligibility");
    auto new_evidence = cold_evidence(1);
    new_evidence.eligible = false;
    check(fixture.scope->publish_evidence(replacement.handle(), new_evidence),
          "new generation could not restart its evidence revision");
    fixture.step(8);
    check(bool(fixture.repo->try_pin(replacement.handle())), "new protected generation was retired");
    replacement.reset();
    replacements.clear();
}

static void malformed_evidence_is_rejected() {
    Fixture fixture(HistoricalGcMode::Off);
    fixture.append();
    std::vector<HistoricalEvidence> invalid;
    auto e = cold_evidence(0); invalid.push_back(e);
    e = cold_evidence(); e.cold_observations = e.observations + 1; invalid.push_back(e);
    e = cold_evidence(); e.importance = std::numeric_limits<double>::quiet_NaN(); invalid.push_back(e);
    e = cold_evidence(); e.importance = 1.1; invalid.push_back(e);
    e = cold_evidence(); e.model_reclaim_score = -0.1; invalid.push_back(e);
    for (const auto &bad : invalid) {
        bool rejected = false;
        try { fixture.scope->publish_evidence(fixture.handles[0], bad); }
        catch (const std::invalid_argument &) { rejected = true; }
        check(rejected, "native evidence API accepted invalid evidence");
    }
}

int main() {
    try {
        selection_holds_are_a_mechanism_veto();
        selection_failure_close_and_scope_reuse_are_safe();
        selection_admission_races_native_retirement();
        statistical_evidence_is_required_and_revision_qualified();
        changed_evidence_vetoes_a_delayed_statistical_intent();
        evidence_revision_changes_force_a_fresh_decision();
        statistical_evidence_does_not_cross_generation_reuse();
        malformed_evidence_is_rejected();
        lru_retirement_and_cpu_reclamation_do_not_wait_for_gpu();
        completions_are_reliable_until_acknowledged_and_across_reuse();
        off_and_shadow_do_not_change_real_history();
        proactive_policy_can_delete_unsealed_active_blocks_without_pressure();
        policy_failure_and_pause_do_not_stop_committed_releases();
        original_io_can_publish_after_retirement_without_resurrection();
        foreign_owners_and_overwrites_are_rejected_and_scope_slots_recycle();
        background_manager_progresses_without_foreground_policy_calls();
        store_can_finish_and_stage_out_a_retired_gpu_window();
        engine_metadata_does_not_contend_with_io_publication();
        execution_admission_races_retirement_and_generation_reuse();
        policy_frame_holds_revalidate_old_decisions_and_release_without_cpu_pins();
        policy_frame_hold_is_optional_for_the_retirement_mechanism();
        failed_frame_admission_rolls_back_earlier_policy_holds();
        requested_capacity_can_exceed_static_lru_watermarks();
        scope_cpu_target_is_independent_of_global_free_bytes();
        serving_metadata_is_preallocated_and_recycled_before_scope_ack();
        serving_metadata_is_preallocated_and_recycled_before_scope_ack(false);
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    std::cout << "Historical KV native manager tests passed\n";
}
