// Opt-in intra-call parallelism for independent publication rows.
//
// KVMEM_NATIVE_PUBLISH_THREADS=N (1..64, default 1) lets a native publication
// call split its already-validated, mutually independent rows (layers or
// records) across N threads: the caller plus N-1 persistent workers. Each row
// keeps its original scalar arithmetic and order, so results are bitwise equal
// to the serial loop. Exceptions are captured per row and the first one in row
// order is rethrown after every row has stopped, matching "nothing returned".
// Callers must hold no Python object references inside fn (run without GIL).
// A concurrent second caller never waits for the busy pool: it runs serially.
#pragma once

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <functional>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace kvmem_parallel {

inline size_t configured_threads() {
    const char *value = std::getenv("KVMEM_NATIVE_PUBLISH_THREADS");
    if (!value || !*value) return 1;
    char *end = nullptr;
    const long parsed = std::strtol(value, &end, 10);
    if (end == value || *end || parsed < 1 || parsed > 64)
        throw std::invalid_argument("KVMEM_NATIVE_PUBLISH_THREADS must be an integer in [1, 64]");
    // Portable: never oversubscribe a small machine. hardware_concurrency()
    // may report 0 when unknown; results are identical for any thread count.
    const size_t cores = std::thread::hardware_concurrency();
    const size_t requested = static_cast<size_t>(parsed);
    return cores ? std::min(requested, cores) : requested;
}

class Pool {
public:
    static Pool &get() {
        // Intentionally leaked: workers block forever on the condition and must
        // not be joined/destroyed during interpreter or static teardown.
        static Pool *pool = new Pool();
        return *pool;
    }

    // Returns false without running anything when another batch owns the pool.
    bool try_run(size_t n, size_t threads, const std::function<void(size_t)> &fn,
                 std::vector<std::exception_ptr> &errors) {
        std::unique_lock<std::mutex> owner(busy_, std::try_to_lock);
        if (!owner.owns_lock()) return false;
        const size_t helpers = threads - 1;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            while (workers_.size() < helpers) {
                const size_t id = workers_.size();
                workers_.emplace_back([this, id] { loop(id); });
                workers_.back().detach();
            }
            fn_ = &fn;
            errors_ = &errors;
            n_ = n;
            wanted_ = helpers;
            active_ = helpers;
            next_.store(0);
            ++generation_;
        }
        wake_.notify_all();
        drain(fn, errors, n);
        std::unique_lock<std::mutex> lock(mutex_);
        done_.wait(lock, [this] { return active_ == 0; });
        fn_ = nullptr;
        errors_ = nullptr;
        return true;
    }

    size_t workers() {
        std::lock_guard<std::mutex> lock(mutex_);
        return workers_.size();
    }

private:
    void drain(const std::function<void(size_t)> &fn, std::vector<std::exception_ptr> &errors, size_t n) {
        for (size_t i; (i = next_.fetch_add(1)) < n;) {
            try {
                fn(i);
            } catch (...) {
                errors[i] = std::current_exception();
            }
        }
    }

    void loop(size_t id) {
        uint64_t seen = 0;
        for (;;) {
            const std::function<void(size_t)> *fn;
            std::vector<std::exception_ptr> *errors;
            size_t n;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                wake_.wait(lock, [&] { return generation_ != seen; });
                seen = generation_;
                // A generation cannot advance until every wanted helper has
                // decremented active_, so a participating helper never skips one.
                if (id >= wanted_) continue;
                fn = fn_;
                errors = errors_;
                n = n_;
            }
            drain(*fn, *errors, n);
            {
                std::lock_guard<std::mutex> lock(mutex_);
                if (--active_ == 0) done_.notify_all();
            }
        }
    }

    std::mutex busy_, mutex_;
    std::condition_variable wake_, done_;
    std::vector<std::thread> workers_;
    const std::function<void(size_t)> *fn_ = nullptr;
    std::vector<std::exception_ptr> *errors_ = nullptr;
    size_t n_ = 0, wanted_ = 0, active_ = 0;
    uint64_t generation_ = 0;
    std::atomic<size_t> next_{0};
};

// Runs fn(0..n-1); serial when disabled, n < 2 or the pool is busy. Returns
// the number of threads that were offered the rows (1 means serial).
inline size_t for_rows(size_t n, const std::function<void(size_t)> &fn) {
    size_t threads = configured_threads();
    if (threads > n) threads = n;
    if (threads > 1) {
        std::vector<std::exception_ptr> errors(n);
        if (Pool::get().try_run(n, threads, fn, errors)) {
            for (auto &error : errors)
                if (error) std::rethrow_exception(error);
            return threads;
        }
    }
    for (size_t i = 0; i < n; ++i) fn(i);
    return 1;
}

}  // namespace kvmem_parallel
