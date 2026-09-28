// Host-only bindings. Original algorithms remain in third_party/qw3.
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "qw3/kvmem_store.hpp"
#include "qw3/nvme_kv_tier.hpp"
#include "qw3/kvmem_repository_gc.hpp"

namespace py = pybind11;
using namespace qw3;

// vector<move-only Lease> can falsely satisfy is_copy_constructible on libstdc++.
// Give pybind11 an explicitly non-copyable owner for asynchronous read leases.
struct OwnedRead : KvMemStorageRead {
    explicit OwnedRead(KvMemStorageRead &&r) : KvMemStorageRead(std::move(r)) {}
    OwnedRead(const OwnedRead &) = delete;
    OwnedRead(OwnedRead &&) = default;
};

// Python callbacks run only on the CPU policy worker. Joining it while holding
// the GIL would deadlock a pending release callback. Store's keep_alive binding
// below retains this bridge as long as native Store retains its GC instance.
struct RepositoryGcBridge {
    py::function release;
    std::shared_ptr<KvMemRepositoryGc> native;
    std::mutex stop_mutex, error_mutex;
    std::string callback_error;

    RepositoryGcBridge(KvMemRepositoryGc::Config config,
                       std::shared_ptr<KvMemRepository> repository, py::function callback)
        : release(std::move(callback)) {
        native = std::make_shared<KvMemRepositoryGc>(config, std::move(repository),
            [this](const KvMemRepository::Record &record) {
                py::gil_scoped_acquire gil;
                try {
                    return release(py::cast(record, py::return_value_policy::copy)).cast<bool>();
                } catch (py::error_already_set &error) {
                    const std::string message = error.what();
                    {
                        std::lock_guard<std::mutex> guard(error_mutex);
                        if (callback_error.empty()) callback_error = message;
                    }
                    // The native worker records failures and retains Releasing
                    // generations for retry. Surface the error on the foreground.
                    throw std::runtime_error(message);
                }
            });
    }
    void stop() {
        py::gil_scoped_release no_gil;
        std::lock_guard<std::mutex> guard(stop_mutex);
        native->stop();
    }
    ~RepositoryGcBridge() {
        stop();
        native.reset();  // Callback and py::function destruct with the GIL held.
    }
    std::string last_error() {
        std::lock_guard<std::mutex> guard(error_mutex);
        return callback_error;
    }
};

PYBIND11_MODULE(_native, m) {
    py::class_<KvMemGcConfig>(m, "GcPolicy")
        .def(py::init<>())
#define GCP(x) .def_readwrite(#x, &KvMemGcConfig::x)
        GCP(max_live_blocks) GCP(half_life_steps) GCP(grace_steps)
        GCP(candidate_steps) GCP(low_observations) GCP(scan_blocks) GCP(batch_blocks)
        GCP(poll_ms) GCP(worker_cpu) GCP(shadow) GCP(trace)
#undef GCP
        ;
    py::enum_<KvMemRepositoryGc::Mode>(m, "GcMode")
        .value("OFF", KvMemRepositoryGc::Mode::Off)
        .value("SHADOW", KvMemRepositoryGc::Mode::Shadow)
        .value("DELETE", KvMemRepositoryGc::Mode::Delete);
    py::class_<KvMemRepositoryGc::Config>(m, "RepositoryGcConfig")
        .def(py::init<>())
#define GCC(x) .def_readwrite(#x, &KvMemRepositoryGc::Config::x)
        GCC(policy) GCC(ready_low) GCC(ready_high) GCC(recent_tokens)
        GCC(candidate_ingest_tokens) GCC(tick_budget_us) GCC(mode) GCC(background)
#undef GCC
        ;
    py::class_<RepositoryGcBridge, std::shared_ptr<RepositoryGcBridge>>(m, "RepositoryGc")
        .def(py::init<KvMemRepositoryGc::Config, std::shared_ptr<KvMemRepository>, py::function>())
        .def("stop", &RepositoryGcBridge::stop)
        .def("poll", [](RepositoryGcBridge &gc) {
            py::gil_scoped_release no_gil;
            gc.native->poll();
        })
        .def("notify_progress", [](RepositoryGcBridge &gc) { gc.native->notify_progress(); })
        .def("begin_step", [](RepositoryGcBridge &gc) { return gc.native->begin_step(); })
        .def("set_history_end", [](RepositoryGcBridge &gc, uint64_t end) { gc.native->set_history_end(end); })
        .def("publish", [](RepositoryGcBridge &gc, const KvMemRepository::Lease &lease, uint32_t flags, bool used) {
            gc.native->publish(lease, flags, used);
        }, py::arg("lease"), py::arg("flags"), py::arg("used") = false)
        .def_property_readonly("last_error", &RepositoryGcBridge::last_error)
        .def_property_readonly("stats", [](RepositoryGcBridge &gc) {
            const auto s = gc.native->stats(); py::dict d;
#define GCS(x) d[#x] = s.x
            GCS(ticks); GCS(examined); GCS(retired); GCS(reclaimed); GCS(protected_blocks);
            GCS(deferred); GCS(failures); GCS(raced_updates); GCS(score_gaps); GCS(stale_scores);
            GCS(shadow_candidates); GCS(shadow_reuses); GCS(worker_ns); GCS(max_tick_ns);
            GCS(metadata_bytes); GCS(policy_fallback_retired); GCS(incomplete_backing); GCS(reader_conflicts);
#undef GCS
            return d;
        });
#define GCF(label, value) m.attr(label) = py::int_(uint32_t(value))
    GCF("GC_REGISTERED", KvMemGc::Registered); GCF("GC_SEALED", KvMemGc::Sealed);
    GCF("GC_GPU", KvMemGc::Gpu); GCF("GC_IO", KvMemGc::Io);
    GCF("GC_VALUE_BACKED", KvMemGc::ValueBacked); GCF("GC_RAW_BACKED", KvMemGc::RawBacked);
    GCF("GC_ANCHOR", KvMemGc::Anchor); GCF("GC_MANDATORY", KvMemGc::Mandatory);
    GCF("GC_CPU_CACHED", KvMemGc::CpuCached); GCF("GC_INDEXED", KvMemRepositoryGc::Indexed);
#undef GCF
    m.def("resolve_keep_bands", [](uint32_t block_tokens, uint32_t select_budget,
                                   int64_t sink_blocks, int64_t recent_blocks) {
        const auto bands = resolve_kvmem_keep_allocation(
            block_tokens, select_budget, sink_blocks, recent_blocks, -1, -1);
        return py::make_tuple(bands.sink_blocks, bands.recent_blocks);
    });
    py::class_<KvMemBlockHandle>(m, "Handle")
        .def(py::init<uint32_t, uint64_t>())
        .def_readonly("slot", &KvMemBlockHandle::slot)
        .def_readonly("generation", &KvMemBlockHandle::generation)
        .def("__eq__", &KvMemBlockHandle::operator==);
    py::class_<KvMemRepository::Record>(m, "Record")
        .def_readonly("handle", &KvMemRepository::Record::handle)
        .def_readonly("logical_id", &KvMemRepository::Record::logical_id)
        .def_readonly("token_begin", &KvMemRepository::Record::token_begin)
        .def_readonly("token_count", &KvMemRepository::Record::token_count);
    py::class_<KvMemRepository::Lease>(m, "Lease")
        .def("__bool__", [](const KvMemRepository::Lease &x) { return bool(x); })
        .def_property_readonly("handle", &KvMemRepository::Lease::handle)
        .def_property_readonly("record", &KvMemRepository::Lease::record,
                               py::return_value_policy::copy)
        .def("clone", &KvMemRepository::Lease::clone)
        .def("reset", &KvMemRepository::Lease::reset);
    py::class_<KvMemRepository::Reservation>(m, "Reservation")
        .def("__bool__", [](const KvMemRepository::Reservation &x) { return bool(x); })
        .def_property_readonly("record", &KvMemRepository::Reservation::record,
                               py::return_value_policy::copy)
        .def("lease", &KvMemRepository::Reservation::lease)
        .def("commit", &KvMemRepository::Reservation::commit)
        .def("cancel", &KvMemRepository::Reservation::cancel);
    py::enum_<KvMemRepository::ReclaimResult>(m, "ReclaimResult")
        .value("RECLAIMED", KvMemRepository::ReclaimResult::Reclaimed)
        .value("READERS", KvMemRepository::ReclaimResult::DeferredReaders)
        .value("RELEASE", KvMemRepository::ReclaimResult::DeferredRelease)
        .value("NOT_RETIRED", KvMemRepository::ReclaimResult::NotRetired)
        .value("STALE", KvMemRepository::ReclaimResult::StaleHandle)
        .value("BUSY", KvMemRepository::ReclaimResult::Busy);
    py::class_<KvMemRepository, std::shared_ptr<KvMemRepository>>(m, "Repository")
        .def(py::init([](uint32_t capacity, uint32_t block_tokens) {
            return std::make_shared<KvMemRepository>(KvMemRepository::Config{capacity, block_tokens});
        }))
        .def("reserve", &KvMemRepository::try_reserve)
        .def("pin", &KvMemRepository::try_pin)
        .def("pin_slot", &KvMemRepository::try_pin_slot)
        .def("retire", &KvMemRepository::retire)
        .def("retired_at", &KvMemRepository::retired_at)
        .def("reclaim", [](KvMemRepository &r, KvMemBlockHandle h, py::function release) {
            return r.try_reclaim(h, [&](const KvMemRepository::Record &record) {
                return release(py::cast(record)).cast<bool>();
            });
        })
        .def_property_readonly("ready_slots", &KvMemRepository::ready_slots)
        .def_property_readonly("stats", [](const KvMemRepository &r) {
            auto s = r.stats(); py::dict d;
#define STAT(x) d[#x] = s.x
            STAT(capacity); STAT(ready); STAT(writing); STAT(live); STAT(retired);
            STAT(releasing); STAT(cancelled); STAT(leases); STAT(reservations);
            STAT(reclaimed); STAT(metadata_bytes);
#undef STAT
            return d;
        });
    py::class_<KvMemStoreConfig>(m, "StoreConfig")
        .def(py::init<>())
#define CFG(x) .def_readwrite(#x, &KvMemStoreConfig::x)
        CFG(block_tokens) CFG(select_budget) CFG(prefill_budget) CFG(gen_budget)
        CFG(sink_blocks) CFG(recent_blocks) CFG(repository_blocks)
        CFG(immutable_source_k) CFG(immutable_refresh_abs_delta_tokens)
        CFG(mtp_enabled)
#undef CFG
        ;
    py::class_<KvMemStorageSpan>(m, "StorageSpan")
        .def_readonly("history_begin", &KvMemStorageSpan::history_begin)
        .def_readonly("storage_begin", &KvMemStorageSpan::storage_begin)
        .def_readonly("token_count", &KvMemStorageSpan::token_count)
        .def_readonly("handle", &KvMemStorageSpan::handle);
    py::class_<OwnedRead>(m, "StorageRead")
        .def_property_readonly("spans", [](const OwnedRead &r) { return r.spans; });
    py::class_<KvMemRemap>(m, "Remap")
#define REMAP(x) .def_readonly(#x, &KvMemRemap::x)
        REMAP(block_id) REMAP(n_tokens) REMAP(from_base) REMAP(to_base)
        REMAP(skip) REMAP(raw_refresh) REMAP(working_k_resident)
#undef REMAP
        ;
    py::class_<KvMemPlan>(m, "Plan")
#define PLAN(x) .def_readonly(#x, &KvMemPlan::x)
        PLAN(stage_in) PLAN(stage_out) PLAN(remaps) PLAN(total_window_tokens)
        PLAN(selection_overlap_blocks) PLAN(gpu_reused_blocks)
        PLAN(retained_position_stable) PLAN(retained_position_moved)
#undef PLAN
        ;
    py::class_<KvMemStore::RepositoryAppend>(m, "Append")
        .def_property_readonly("spans", &KvMemStore::RepositoryAppend::spans)
        .def_property_readonly("begin", &KvMemStore::RepositoryAppend::begin)
        .def_property_readonly("end", &KvMemStore::RepositoryAppend::end)
        .def("commit", &KvMemStore::RepositoryAppend::commit)
        .def("storage_write", [](const KvMemStore::RepositoryAppend &a, uint64_t b, uint32_t n) {
            return OwnedRead(a.storage_write(b, n));
        })
        .def("discarded_page_ranges", &KvMemStore::RepositoryAppend::discarded_page_ranges);
    py::class_<KvMemStore>(m, "Store")
        .def(py::init<KvMemStoreConfig, std::shared_ptr<KvMemRepository>>())
        .def("attach_gc", [](KvMemStore &store, std::shared_ptr<RepositoryGcBridge> gc) {
            py::gil_scoped_release no_gil;
            store.attach_repository_gc(gc ? gc->native : nullptr);
        }, py::keep_alive<1, 2>())
        .def("gc_begin_step", &KvMemStore::gc_begin_step)
        .def("gc_end_step", &KvMemStore::gc_end_step)
        .def("gc_flags", &KvMemStore::set_repository_gc_flags)
        .def("working_resident", [](KvMemStore &store, KvMemBlockHandle handle, bool resident) {
            // The plugin owns archive locations; no fictitious native CPU/SSD
            // slot is installed. Durable raw/V/index flags are independent.
            store.set_block_tier(handle.slot, resident ? KvTier::GPU : KvTier::SSD,
                                 -1, -1, handle.generation);
        })
        .def("prepare_append", &KvMemStore::prepare_repository_append, py::keep_alive<0, 1>())
        .def("read", [](const KvMemStore &s, uint64_t b, uint32_t n) {
            return OwnedRead(s.repository_storage_read(b, n));
        })
        .def("handle", &KvMemStore::repository_handle)
        .def_property_readonly("history_end", &KvMemStore::history_end)
        .def("scores", &KvMemStore::set_repository_retrieval_scores)
        .def("pick", py::overload_cast<const std::vector<uint32_t>&>(
                         &KvMemStore::pick_topk_blocks, py::const_))
        .def("select", &KvMemStore::set_repository_selection,
             py::arg("handles"), py::arg("force_raw_refresh") = false)
        .def("set_baked", &KvMemStore::set_block_baked_pos,
             py::arg("slot"), py::arg("position"), py::arg("generation"));

    py::class_<NvmeAllocation>(m, "NvmeAllocation")
        .def_readonly("block_id", &NvmeAllocation::block_id)
        .def_readonly("slot", &NvmeAllocation::slot)
        .def_readonly("generation", &NvmeAllocation::generation);
    py::class_<NvmeIoSpan>(m, "IoSpan")
        .def(py::init<int32_t, uint64_t, uint64_t>());
    py::class_<NvmeKvTier>(m, "NvmeTier")
        .def(py::init([](const std::string &dir, uint64_t total, uint64_t slot) {
            if (slot == 0 || total < slot || total / slot > UINT32_MAX)
                throw std::invalid_argument("Invalid NVMe arena capacity");
            NvmeKvTierConfig cfg;
            cfg.dir = dir; cfg.total_bytes = total; cfg.slot_bytes = slot;
            cfg.drop_page_cache = true;
            return std::make_unique<NvmeKvTier>(cfg);
        }))
        .def("allocate", [](NvmeKvTier &tier, uint32_t block) {
            tier.place_block(block);
            return tier.allocation(block);
        })
        .def("release", &NvmeKvTier::try_release_allocation)
        .def_property_readonly("used_slots", &NvmeKvTier::used_slots)
        .def("io", [](NvmeKvTier &tier, const std::vector<NvmeIoSpan> &spans,
                       py::buffer buffer, bool write) {
            auto info = buffer.request(!write);
            if (info.ndim != 1 || info.itemsize != 1 || info.strides[0] != 1)
                throw std::invalid_argument("NVMe IO requires contiguous one-dimensional bytes");
            NvmeBatchIoStats stats;
            {
                py::gil_scoped_release release;
                if (write) tier.write_spans(spans, info.ptr, info.size, &stats);
                else tier.read_spans(spans, info.ptr, info.size, &stats);
            }
            py::dict result;
            result["bytes"] = stats.bytes; result["syscalls"] = stats.syscalls;
            result["duration_ns"] = stats.duration_ns;
            return result;
        });
}
