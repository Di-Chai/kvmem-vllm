// Session tiering (cpu_session_ttl_ms): batched CPU <-> SSD record I/O without the GIL.
//
// The Python tier keeps every state decision (eligibility, extent allocation,
// the payload spill/restore flip, bookkeeping) under its own lock; these
// helpers only move bytes for a whole batch of records: one GIL release per
// batch instead of several per record. Records are verified with XXH64.
#pragma once

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <stdexcept>
#include <system_error>
#include <unistd.h>
#include <vector>

#include "historical_payload.hpp"

namespace vllm_kvmem::session_spill {

namespace detail {
constexpr uint64_t P1 = 11400714785074694791ULL, P2 = 14029467366897019727ULL,
                   P3 = 1609587929392839161ULL, P4 = 9650029242287828579ULL,
                   P5 = 2870177450012600261ULL;
inline uint64_t rotl(uint64_t x, int r) { return (x << r) | (x >> (64 - r)); }
inline uint64_t read64(const uint8_t *p) { uint64_t v; std::memcpy(&v, p, 8); return v; }
inline uint64_t read32(const uint8_t *p) { uint32_t v; std::memcpy(&v, p, 4); return v; }
inline uint64_t round(uint64_t acc, uint64_t input) { return rotl(acc + input * P2, 31) * P1; }
inline uint64_t merge(uint64_t acc, uint64_t value) { return (acc ^ round(0, value)) * P1 + P4; }

inline void advise_drop(int fd, uint64_t offset, size_t bytes) {
#if defined(POSIX_FADV_DONTNEED)
    ::posix_fadvise(fd, static_cast<off_t>(offset), static_cast<off_t>(bytes), POSIX_FADV_DONTNEED);
#else
    (void)fd; (void)offset; (void)bytes;
#endif
}

inline void write_all(int fd, const uint8_t *data, size_t bytes, uint64_t offset) {
    while (bytes) {
        const ssize_t n = ::pwrite(fd, data, bytes, static_cast<off_t>(offset));
        if (n < 0) {
            if (errno == EINTR) continue;
            throw std::system_error(errno, std::generic_category(), "Session spill write failed");
        }
        if (n == 0) throw std::runtime_error("Session spill write made no progress");
        data += n; bytes -= static_cast<size_t>(n); offset += static_cast<uint64_t>(n);
    }
}

inline void read_all(int fd, uint8_t *data, size_t bytes, uint64_t offset) {
    while (bytes) {
        const ssize_t n = ::pread(fd, data, bytes, static_cast<off_t>(offset));
        if (n < 0) {
            if (errno == EINTR) continue;
            throw std::system_error(errno, std::generic_category(), "Session promotion read failed");
        }
        if (n == 0) throw std::runtime_error("Spilled record is truncated");
        data += n; bytes -= static_cast<size_t>(n); offset += static_cast<uint64_t>(n);
    }
}
}  // namespace detail

// XXH64 (xxHash, 64-bit, seed 0). Native byte order: digests never leave the host.
inline uint64_t xxh64(const void *data, size_t bytes) {
    using namespace detail;
    const auto *p = static_cast<const uint8_t *>(data);
    const uint8_t *const end = p + bytes;
    uint64_t h;
    if (bytes >= 32) {
        uint64_t v1 = P1 + P2, v2 = P2, v3 = 0, v4 = 0 - P1;
        const uint8_t *const limit = end - 32;
        do {
            v1 = round(v1, read64(p)); v2 = round(v2, read64(p + 8));
            v3 = round(v3, read64(p + 16)); v4 = round(v4, read64(p + 24));
            p += 32;
        } while (p <= limit);
        h = rotl(v1, 1) + rotl(v2, 7) + rotl(v3, 12) + rotl(v4, 18);
        h = merge(h, v1); h = merge(h, v2); h = merge(h, v3); h = merge(h, v4);
    } else {
        h = P5;
    }
    h += static_cast<uint64_t>(bytes);
    for (; p + 8 <= end; p += 8) h = rotl(h ^ round(0, read64(p)), 27) * P1 + P4;
    if (p + 4 <= end) { h = rotl(h ^ (read32(p) * P1), 23) * P2 + P3; p += 4; }
    for (; p < end; ++p) h = rotl(h ^ (static_cast<uint64_t>(*p) * P5), 11) * P1;
    h ^= h >> 33; h *= P2; h ^= h >> 29; h *= P3; h ^= h >> 32;
    return h;
}

using Arena = HistoricalPayloadArena;

// Copy each Live record to (fds[i], offsets[i]) and digest it. written[i] = 0
// when the record has no CPU bytes (stale, spilled or retiring). The payload
// is not dropped here: the caller flips it under its own lock (spill()).
inline void write_records(const Arena &arena, const Arena::Handle *handles, const int *fds,
                          const uint64_t *offsets, size_t count, uint8_t *written, uint64_t *digests,
                          bool drop_cache) {
    const size_t bytes = arena.record_bytes();
    for (size_t i = 0; i < count; ++i) {
        written[i] = 0;
        digests[i] = 0;
        auto read = arena.acquire(handles[i]);
        if (!read) continue;
        if (read.size() != bytes) throw std::invalid_argument("Spilled record size differs from the arena");
        detail::write_all(fds[i], read.data(), bytes, offsets[i]);
        digests[i] = xxh64(read.data(), bytes);
        written[i] = 1;
        if (drop_cache) detail::advise_drop(fds[i], offsets[i], bytes);   // starts write-back now
    }
}

// Read, verify and restore each still-spilled record. restored[i] = 0 when
// another reader already restored it (or it was released): its extent may
// then have been reused, so a checksum mismatch is fatal only while the
// record is still spilled.
inline void read_records(Arena &arena, const Arena::Handle *handles, const int *fds, const uint64_t *offsets,
                         const uint64_t *digests, size_t count, uint8_t *restored, bool drop_cache) {
    const size_t bytes = arena.record_bytes();
    thread_local std::vector<uint8_t> buffer;
    if (buffer.size() < bytes) buffer.resize(bytes);
    for (size_t i = 0; i < count; ++i) {
        restored[i] = 0;
        if (!arena.spilled(handles[i])) continue;
        detail::read_all(fds[i], buffer.data(), bytes, offsets[i]);
        if (drop_cache) detail::advise_drop(fds[i], offsets[i], bytes);
        if (xxh64(buffer.data(), bytes) != digests[i]) {
            if (!arena.spilled(handles[i])) continue;
            throw std::runtime_error("Spilled record checksum mismatch");
        }
        const Arena::Segment segment{buffer.data(), bytes};
        restored[i] = arena.restore(handles[i], &segment, 1) ? 1 : 0;
    }
}

// Wait until the written ranges are on the device, then drop them from the
// page cache. A file closed concurrently (its records were released) is skipped.
inline void flush_ranges(const int *fds, const uint64_t *offsets, size_t count, size_t bytes) {
    std::vector<int> unique(fds, fds + count);
    std::sort(unique.begin(), unique.end());
    unique.erase(std::unique(unique.begin(), unique.end()), unique.end());
    for (int fd : unique) {
#if defined(__APPLE__)
        const int rc = ::fsync(fd);
#else
        const int rc = ::fdatasync(fd);
#endif
        if (rc != 0 && errno != EBADF && errno != EINVAL && errno != EINTR)
            throw std::system_error(errno, std::generic_category(), "Session spill flush failed");
    }
    for (size_t i = 0; i < count; ++i) detail::advise_drop(fds[i], offsets[i], bytes);
}

}  // namespace vllm_kvmem::session_spill
