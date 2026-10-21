// Private outputs for bounded serial allocation and disjoint parallel memcpy.
// Nothing implements the buffer protocol or exposes bytes before completion.
#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

namespace vllm_kvmem {
namespace py = pybind11;

class PrivateRecordPack {
public:
    PrivateRecordPack(const std::vector<py::buffer> &sources, size_t record_bytes,
                      size_t max_payload_bytes) : record_bytes_(record_bytes) {
        if (sources.empty() || sources.size() % 2 || !record_bytes ||
                record_bytes > static_cast<size_t>(PY_SSIZE_T_MAX) ||
                sources.size()/2 > max_payload_bytes/record_bytes)
            throw std::invalid_argument("Private record pack exceeds its admitted payload budget");
        // Validate every export before allocating any destination.
        size_t pair_bytes = 0;
        views_.reserve(sources.size());
        for (const auto &source : sources) {
            auto view = source.request();
            if (view.ndim != 1 || view.itemsize != 1 || view.strides[0] != 1 ||
                    view.size < 0 || static_cast<size_t>(view.size) > record_bytes-pair_bytes)
                throw std::invalid_argument("Invalid private record pair view");
            pair_bytes += static_cast<size_t>(view.size);
            views_.push_back(std::move(view));
            if (views_.size() % 2 == 0) {
                if (pair_bytes != record_bytes)
                    throw std::invalid_argument("Private record pair size mismatch");
                pair_bytes = 0;
            }
        }
        const auto count = sources.size()/2;
        records_.reserve(count);
        destinations_.reserve(count);
        states_.assign(count, 0);
        for (size_t i = 0; i < count; ++i) {
            auto *allocated = PyBytes_FromStringAndSize(nullptr, static_cast<Py_ssize_t>(record_bytes));
            if (!allocated) throw py::error_already_set();
            records_.push_back(py::reinterpret_steal<py::bytes>(allocated));
            destinations_.push_back(PyBytes_AS_STRING(records_.back().ptr()));
        }
    }
    PrivateRecordPack(const PrivateRecordPack &) = delete;

    size_t copy_range(size_t begin, size_t end) {
        // All state transitions run with the GIL. The memcpy interval only
        // touches disjoint output ranges; close/finish reject active writers.
        if (closed_ || begin >= end || end > states_.size())
            throw std::invalid_argument("Invalid private record copy range or closed batch");
        for (size_t i = begin; i < end; ++i)
            if (states_[i]) throw std::logic_error("Private record range already claimed");
        for (size_t i = begin; i < end; ++i) states_[i] = 1;
        ++active_;
        {
            py::gil_scoped_release release;
            for (size_t i = begin; i < end; ++i) {
                std::memcpy(destinations_[i], views_[2*i].ptr, views_[2*i].size);
                std::memcpy(destinations_[i]+views_[2*i].size, views_[2*i+1].ptr,
                            views_[2*i+1].size);
            }
        }
        for (size_t i = begin; i < end; ++i) states_[i] = 2;
        --active_;
        return (end-begin)*record_bytes_;
    }

    std::vector<py::bytes> finish() {
        if (closed_ || active_)
            throw std::logic_error("Private record pack is closed or has active writers");
        for (auto state : states_)
            if (state != 2) throw std::logic_error("Private record pack is incomplete");
        closed_ = true;
        views_.clear();
        destinations_.clear();
        return std::move(records_);
    }

    bool abort() {
        if (active_) return false;
        if (!closed_) {
            closed_ = true;
            views_.clear();
            destinations_.clear();
            records_.clear();
        }
        return true;
    }

private:
    size_t record_bytes_, active_ = 0;
    bool closed_ = false;
    std::vector<py::buffer_info> views_;
    std::vector<py::bytes> records_;
    std::vector<char *> destinations_;
    std::vector<uint8_t> states_;
};
} // namespace vllm_kvmem
