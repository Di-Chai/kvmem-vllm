// Host-only bindings. Original algorithms remain in third_party/qw3.
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "qw3/kvmem_store.hpp"
#include "qw3/nvme_kv_tier.hpp"
#include "qw3/kvmem_repository_gc.hpp"
#include <filesystem>
#include <cstring>
#include <limits>
#include <sys/stat.h>

namespace py = pybind11;
using namespace qw3;

// A private, named inode allows strict reservation verification without changing
// the imported tier. Unlink at construction exit (including exceptions), leaving
// the tier's own fd as the sole owner, exactly as for its ephemeral arena.
struct ReservationFile {
    int fd = -1;
    std::string path;
    explicit ReservationFile(const std::string &directory) {
        std::filesystem::create_directories(directory);
        std::string pattern = directory + "/kvmem-reserve-XXXXXX";
        std::vector<char> name(pattern.begin(), pattern.end());
        name.push_back('\0');
        fd = ::mkstemp(name.data());
        if (fd < 0) throw std::runtime_error("Cannot create private NVMe reservation file");
        path = name.data();
    }
    ~ReservationFile() {
        if (fd >= 0) ::close(fd);
        if (!path.empty()) ::unlink(path.c_str());
    }
    ReservationFile(const ReservationFile &) = delete;
    ReservationFile &operator=(const ReservationFile &) = delete;
};

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

#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
#define KVMEM_AVX2 __attribute__((target("avx2")))
static bool native_decode_sum_supported() { return __builtin_cpu_supports("avx2"); }
#else
#define KVMEM_AVX2
static bool native_decode_sum_supported() { return false; }
#endif

static inline float bf16_as_float(const char *source) {
    uint16_t bf16;
    std::memcpy(&bf16, source, 2);
    const uint32_t bits = static_cast<uint32_t>(bf16) << 16;
    float value;
    std::memcpy(&value, &bits, 4);
    return value;
}

template<int Count>
KVMEM_AVX2 static void add_decode_mean(const py::buffer_info &raw,
                                     const py::buffer_info &sums, ssize_t begin) {
    const auto width = sums.shape[1] * sums.shape[2];
    auto *out = static_cast<float *>(sums.ptr);
    for (ssize_t layer = 0; layer < sums.shape[0]; ++layer) {
        const auto *base = static_cast<const char *>(raw.ptr) + layer * raw.strides[0] + begin * raw.strides[1];
        auto *destination = out + layer * width;
        for (ssize_t column = 0; column < width; ++column) {
            float total = 0.f;
            // Compile-time count permits vectorization across columns without
            // reassociating the FP32 addition order along the token dimension.
            for (int token = 0; token < Count; ++token)
                total += bf16_as_float(base + token * raw.strides[1] + column * 2);
            destination[column] += total;
        }
    }
}

PYBIND11_MODULE(_native, m) {
    m.def("scatter_history_batch", [](const std::vector<py::tuple> &jobs) {
        struct Row {
            py::buffer_info raw, values, dst_raw, dst_values, delta, sum;
            ssize_t begin, offset, count;
        };
        std::vector<Row> rows;
        rows.reserve(jobs.size());
        for (const auto &job : jobs) {
            if (job.size() != 9) throw std::invalid_argument("Invalid history scatter descriptor");
            Row r{job[0].cast<py::buffer>().request(), job[1].cast<py::buffer>().request(),
                  job[2].cast<py::buffer>().request(), job[3].cast<py::buffer>().request(),
                  job[4].is_none() ? py::buffer_info{} : job[4].cast<py::buffer>().request(),
                  job[5].cast<py::buffer>().request(),
                  job[6].cast<ssize_t>(), job[7].cast<ssize_t>(), job[8].cast<ssize_t>()};
            auto byte_rows = [](const py::buffer_info &v, bool writable) {
                if (v.ndim != 4 || v.itemsize != 1 || (writable && v.readonly)) return false;
                if (v.shape[0] <= 0 || v.shape[1] <= 0 || v.shape[2] <= 0 || v.shape[3] <= 0) return false;
                return v.strides[3] == 1 && v.strides[2] == v.shape[3] &&
                       v.strides[1] >= v.shape[2] * v.shape[3] &&
                       v.strides[0] >= v.shape[2] * v.shape[3];
            };
            auto sums = [](const py::buffer_info &v, bool writable) {
                return v.ndim == 3 && v.format == py::format_descriptor<float>::format() &&
                       v.itemsize == 4 && (!writable || !v.readonly) &&
                       v.strides[2] == 4 && v.strides[1] == v.shape[2] * 4 &&
                       v.strides[0] == v.shape[1] * v.strides[1];
            };
            if (!byte_rows(r.raw, false) || !byte_rows(r.values, false) ||
                !byte_rows(r.dst_raw, true) || !byte_rows(r.dst_values, true) ||
                !sums(r.sum, true) ||
                (r.delta.ptr && (!sums(r.delta, false) || r.delta.shape != r.sum.shape)) ||
                (!r.delta.ptr && (!native_decode_sum_supported() || r.count > 4 ||
                                 (r.sum.shape[1] * r.sum.shape[2]) % 64)) ||
                r.sum.shape[0] <= 0 ||
                r.sum.shape[0] > r.raw.shape[0] || r.sum.shape[1] != r.values.shape[2] ||
                r.sum.shape[2] != r.values.shape[3] || r.begin < 0 || r.offset < 0 || r.count <= 0)
                throw std::invalid_argument("Invalid history scatter geometry or buffer ownership");
            for (ssize_t axis : {0, 2, 3}) {
                if (r.values.shape[axis] != r.dst_values.shape[axis] ||
                    r.raw.shape[axis] != r.dst_raw.shape[axis] ||
                    r.raw.shape[axis] != r.values.shape[axis] * (axis == 3 ? 2 : 1))
                    throw std::invalid_argument("History scatter codec mismatch");
            }
            if (r.raw.shape[1] != r.values.shape[1] ||
                r.dst_raw.shape[1] != r.dst_values.shape[1] ||
                r.count > r.raw.shape[1] || r.begin > r.raw.shape[1] - r.count ||
                r.count > r.dst_raw.shape[1] || r.offset > r.dst_raw.shape[1] - r.count)
                throw std::invalid_argument("History scatter extent outside owned rows");
            rows.push_back(std::move(r));
        }
        // Every descriptor is validated before touching any destination. The
        // exported buffers retain the packet and each authority through this call.
        py::gil_scoped_release no_gil;
        for (const auto &r : rows) {
            for (ssize_t layer = 0; layer < r.raw.shape[0]; ++layer) {
                auto copy_rows = [&](const py::buffer_info &dst, const py::buffer_info &src) {
                    const auto width = src.shape[2] * src.shape[3];
                    auto *out = static_cast<char *>(dst.ptr) + layer * dst.strides[0] + r.offset * dst.strides[1];
                    const auto *in = static_cast<const char *>(src.ptr) + layer * src.strides[0] + r.begin * src.strides[1];
                    if (src.strides[1] == width && dst.strides[1] == width) {
                        std::memcpy(out, in, r.count * width);
                    } else {
                        for (ssize_t token = 0; token < r.count; ++token)
                            std::memcpy(out + token * dst.strides[1], in + token * src.strides[1], width);
                    }
                };
                copy_rows(r.dst_raw, r.raw);
                copy_rows(r.dst_values, r.values);
            }
            auto *sum = static_cast<float *>(r.sum.ptr);
            if (r.delta.ptr) {
                const auto *delta = static_cast<const float *>(r.delta.ptr);
                for (ssize_t i = 0; i < r.sum.size; ++i) sum[i] += delta[i];
            } else {
                // Qualified BF16 decode rows, retaining ATen's FP32 outer
                // reduction order. Larger prefixes use the reference Torch sum.
                switch (r.count) {
                    case 1: add_decode_mean<1>(r.raw, r.sum, r.begin); break;
                    case 2: add_decode_mean<2>(r.raw, r.sum, r.begin); break;
                    case 3: add_decode_mean<3>(r.raw, r.sum, r.begin); break;
                    case 4: add_decode_mean<4>(r.raw, r.sum, r.begin); break;
                }
            }
        }
    });

    m.def("pack_record_views", [](const std::vector<py::buffer> &sources) {
        // Sealed authority tensors stay owned by these exports. Copying their
        // immutable contents on an archive worker must not hold the engine GIL.
        std::vector<py::buffer_info> views;
        size_t total = 0;
        for (const auto &source : sources) {
            auto view = source.request();
            if (view.ndim != 1 || view.itemsize != 1 || view.strides[0] != 1 ||
                    static_cast<size_t>(view.size) > static_cast<size_t>(PY_SSIZE_T_MAX) - total)
                throw std::invalid_argument("Record pack requires bounded contiguous byte views");
            total += static_cast<size_t>(view.size);
            views.push_back(std::move(view));
        }
        if (!total) throw std::invalid_argument("Record pack requires nonempty bytes");
        PyObject *allocated = PyBytes_FromStringAndSize(nullptr, static_cast<Py_ssize_t>(total));
        if (!allocated) throw py::error_already_set();
        auto result = py::reinterpret_steal<py::bytes>(allocated);
        char *destination = PyBytes_AS_STRING(result.ptr());
        {
            py::gil_scoped_release release;
            for (const auto &view : views) {
                std::memcpy(destination, view.ptr, static_cast<size_t>(view.size));
                destination += view.size;
            }
        }
        return result;
    });
    // Each source owns one immutable record. A caller may dispatch disjoint
    // ranges through a bounded persistent worker pool; retain Python owners and
    // the destination buffer export until all memcpy operations have completed.
    // Cache misses are left untouched for the authoritative tier read.
    m.def("gather_cached_records", [](py::buffer destination,
            const std::vector<py::object> &sources, size_t record_bytes,
            size_t begin, size_t end) {
        auto target = destination.request(true);
        if (!record_bytes || begin > end || end > sources.size() ||
                sources.size() > std::numeric_limits<size_t>::max() / record_bytes ||
                target.ndim != 1 || target.itemsize != 1 || target.strides[0] != 1 ||
                static_cast<size_t>(target.size) != sources.size() * record_bytes)
            throw std::invalid_argument("Invalid cached gather byte extent or range");
        std::vector<const char *> pointers;
        pointers.reserve(end - begin);
        for (size_t i = begin; i < end; ++i) {
            if (sources[i].is_none()) pointers.push_back(nullptr);
            else {
                if (!PyBytes_CheckExact(sources[i].ptr()) ||
                        static_cast<size_t>(PyBytes_GET_SIZE(sources[i].ptr())) != record_bytes)
                    throw std::invalid_argument("Cached gather sources must own exact immutable record bytes");
                pointers.push_back(PyBytes_AS_STRING(sources[i].ptr()));
            }
        }
        size_t copied = 0;
        {
            py::gil_scoped_release release;
            auto *out = static_cast<char *>(target.ptr);
            for (size_t i = begin; i < end; ++i) {
                if (const char *source = pointers[i - begin]) {
                    std::memcpy(out + i * record_bytes, source, record_bytes);
                    copied += record_bytes;
                }
            }
        }
        return copied;
    }, py::arg("destination"), py::arg("sources"), py::arg("record_bytes"),
       py::arg("begin"), py::arg("end"));
    m.def("gather_record_region", [](py::buffer destination,
            const std::vector<py::object> &sources, size_t record_bytes,
            size_t offset, size_t length, size_t begin, size_t end) {
        auto target = destination.request(true);
        if (!length || offset > record_bytes || length > record_bytes - offset ||
                begin > end || end > sources.size() ||
                sources.size() > static_cast<size_t>(PY_SSIZE_T_MAX) / length ||
                target.ndim != 1 || target.itemsize != 1 || target.strides[0] != 1 ||
                static_cast<size_t>(target.size) != sources.size() * length)
            throw std::invalid_argument("Invalid record region or destination extent");
        std::vector<py::buffer_info> exports;
        std::vector<const char *> pointers;
        for (size_t i = begin; i < end; ++i) {
            if (sources[i].is_none()) pointers.push_back(nullptr);
            else {
                auto view = py::reinterpret_borrow<py::buffer>(sources[i]).request();
                if (view.ndim != 1 || view.itemsize != 1 || view.strides[0] != 1 ||
                        static_cast<size_t>(view.size) != record_bytes)
                    throw std::invalid_argument("Region source must cover one complete record");
                pointers.push_back(static_cast<const char *>(view.ptr) + offset);
                exports.push_back(std::move(view));
            }
        }
        size_t copied = 0;
        {
            py::gil_scoped_release release;
            for (size_t i = begin; i < end; ++i) {
                if (const char *source = pointers[i - begin]) {
                    std::memcpy(static_cast<char *>(target.ptr) + i * length, source, length);
                    copied += length;
                }
            }
        }
        return copied;
    });
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
        .def("checkpoint", &KvMemStore::repository_checkpoint,
             py::arg("position"), py::arg("pin_working_set") = true)
        .def("checkpoint_valid", &KvMemStore::repository_checkpoint_valid)
        .def("working_records", [](const KvMemStore &store) {
            // Value tuples only: no block/Store views survive into a checkpoint.
            std::vector<std::tuple<uint32_t, uint64_t, uint64_t, uint32_t, int64_t>> records;
            for (const auto &block : store.blocks()) {
                if (block.repository_generation && block.n_tokens &&
                        (block.in_working_set || block.tier == KvTier::GPU))
                    records.emplace_back(block.block_id, block.repository_generation,
                        block.history_pos_start, block.n_tokens, block.baked_pos);
            }
            std::sort(records.begin(), records.end(), [](const auto &a, const auto &b) {
                return std::get<4>(a) < std::get<4>(b);
            });
            return records;
        })
        .def("prepare_rewind", &KvMemStore::prepare_repository_rewind, py::keep_alive<0, 1>())
        .def("read", [](const KvMemStore &s, uint64_t b, uint32_t n) {
            return OwnedRead(s.repository_storage_read(b, n));
        })
        .def("handle", &KvMemStore::repository_handle)
        .def_property_readonly("history_end", &KvMemStore::history_end)
        .def("scores", &KvMemStore::set_repository_retrieval_scores)
        .def("pick", [](const KvMemStore &s, const std::vector<uint32_t> &mandatory) {
            return s.pick_topk_blocks(mandatory);
        })
        .def("select", &KvMemStore::set_repository_selection,
             py::arg("handles"), py::arg("force_raw_refresh") = false)
        .def("set_baked", &KvMemStore::set_block_baked_pos,
             py::arg("slot"), py::arg("position"), py::arg("generation"));

    py::class_<KvMemStore::RepositoryCheckpoint, std::shared_ptr<KvMemStore::RepositoryCheckpoint>>(
            m, "RepositoryCheckpoint")
        .def_property_readonly("position", &KvMemStore::RepositoryCheckpoint::position)
        .def_property_readonly("valid", &KvMemStore::RepositoryCheckpoint::valid);
    py::class_<KvMemStore::RepositoryRewind>(m, "RepositoryRewind")
        .def_property_readonly("position", &KvMemStore::RepositoryRewind::position)
        .def_property_readonly("discarded_spans", &KvMemStore::RepositoryRewind::discarded_spans)
        .def("commit", &KvMemStore::RepositoryRewind::commit);

    py::class_<NvmeAllocation>(m, "NvmeAllocation")
        .def_readonly("block_id", &NvmeAllocation::block_id)
        .def_readonly("slot", &NvmeAllocation::slot)
        .def_readonly("generation", &NvmeAllocation::generation);
    py::class_<NvmeIoSpan>(m, "IoSpan")
        .def(py::init<int32_t, uint64_t, uint64_t>());
    py::class_<NvmeKvTier>(m, "NvmeTier")
        .def(py::init([](const std::string &dir, uint64_t total, uint64_t slot, bool preallocate) {
            if (slot == 0 || total < slot || total / slot > UINT32_MAX)
                throw std::invalid_argument("Invalid NVMe arena capacity");
            py::gil_scoped_release release;
            NvmeKvTierConfig cfg;
            cfg.dir = dir; cfg.total_bytes = total; cfg.slot_bytes = slot;
            cfg.drop_page_cache = true;
            if (preallocate) {
                ReservationFile reservation(dir);
                cfg.file_name = std::filesystem::path(reservation.path).filename().string();
                cfg.durable = true;
                cfg.preallocate = true;
                auto tier = std::make_unique<NvmeKvTier>(cfg);
                struct stat state {};
                if (::fstat(reservation.fd, &state) != 0 || state.st_size < 0 || state.st_blocks < 0 ||
                        uint64_t(state.st_size) < total || uint64_t(state.st_blocks) * 512 < total)
                    throw std::runtime_error("Filesystem did not physically reserve the requested NVMe quota");
                return tier;
            }
            return std::make_unique<NvmeKvTier>(cfg);
        }), py::arg("directory"), py::arg("total_bytes"), py::arg("slot_bytes"), py::arg("preallocate") = false)
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
