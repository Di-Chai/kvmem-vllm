// Host-only transactional prefill feedback. The public Python dictionary is
// replaced only after every row validates and the complete update succeeds.
#pragma once
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <unordered_map>
#include <vector>

namespace kvmem_prefill {
namespace py = pybind11;

// Match CPython 3.12 float sum's compensated positive-float path. Validation
// uses this sum; the mass/exposure updates below deliberately remain ordered
// individual double additions, just like the original Python loop.
struct Sum {
    double hi = 0., lo = 0.;
    void add(double value) {
        const double next = hi + value;
        lo += std::abs(hi) >= std::abs(value) ? (hi-next)+value : (value-next)+hi;
        hi = next;
    }
    double value() const { return lo != 0. && std::isfinite(lo) ? hi+lo : hi; }
};

inline double multiply(double a, double b) {
    // Python multiplies then adds: do not contract into an FMA or reassociate.
    volatile double rounded = a*b;
    return rounded;
}

template<class T>
double read(const py::buffer_info &buffer, ssize_t row, ssize_t column) {
    T value;
    std::memcpy(&value, static_cast<const char *>(buffer.ptr) + row*buffer.strides[0]
                + column*buffer.strides[1], sizeof(value));
    return static_cast<double>(value);
}

inline py::dict commit(const py::dict &previous, double decay,
        const std::vector<int64_t> &rows, const std::vector<int64_t> &starts,
        const std::vector<int64_t> &history, int64_t local_begin, py::buffer masses) {
    const auto buffer = masses.request();
    const bool fp32 = buffer.format == py::format_descriptor<float>::format() && buffer.itemsize == 4;
    const bool fp64 = buffer.format == py::format_descriptor<double>::format() && buffer.itemsize == 8;
    if (buffer.ndim != 2 || (!fp32 && !fp64))
        throw std::invalid_argument("Prefill attention requires a float32/float64 matrix");
    if (buffer.shape[0] != static_cast<ssize_t>(rows.size()))
        throw std::runtime_error("Missing prefill attention rows");
    if (buffer.shape[1] != static_cast<ssize_t>(history.size()))
        throw std::runtime_error("Invalid prefill attention mass");
    if (rows.empty()) throw std::invalid_argument("Prefill attention requires sampled rows");

    std::vector<int64_t> keys;
    std::vector<size_t> indices;
    std::vector<double> mass, exposure;
    std::unordered_map<int64_t, size_t> positions;
    keys.reserve(history.size()); indices.reserve(history.size()); positions.reserve(history.size());
    for (const auto key : history) {
        const auto inserted = positions.emplace(key, keys.size());
        indices.push_back(inserted.first->second);
        if (!inserted.second) continue;
        keys.push_back(key);
        const py::int_ object(key);
        if (previous.contains(object)) {
            const auto old = py::cast<std::vector<double>>(previous[object]);
            if (old.size() != 2) throw std::invalid_argument("Invalid prefill history state");
            mass.push_back(multiply(decay, old[0]));
            exposure.push_back(multiply(decay, old[1]));
        } else {
            mass.push_back(multiply(decay, 0.));
            exposure.push_back(multiply(decay, 0.));
        }
    }
    const auto read_value = fp32 ? &read<float> : &read<double>;
    {
        py::gil_scoped_release release;
        const double inv = 1. / rows.size();
        std::vector<ssize_t> seen(keys.size(), -1);
        for (ssize_t row = 0; row < buffer.shape[0]; ++row) {
            Sum sum;
            for (ssize_t page = 0; page < buffer.shape[1]; ++page) {
                const double value = read_value(buffer, row, page);
                if (!std::isfinite(value) || value < 0.)
                    throw std::runtime_error("Invalid prefill attention mass");
                sum.add(value);
            }
            if (std::abs(sum.value()-1.) > .005)
                throw std::runtime_error("Invalid prefill attention mass");
            // Avoid signed overflow at the native interface; actual window
            // coordinates are bounded by the model context length.
            if ((rows[row] > 0 && local_begin > INT64_MAX-rows[row]) ||
                (rows[row] < 0 && local_begin < INT64_MIN-rows[row]))
                throw std::invalid_argument("Prefill position overflow");
            const auto position = local_begin + rows[row];
            // Python zip truncates at starts; keep that legacy contract.
            for (size_t page = 0; page < std::min(starts.size(), history.size()); ++page) {
                if (starts[page] > position) continue;
                const auto index = indices[page];
                mass[index] += multiply(inv, read_value(buffer, row, page));
                if (seen[index] != row) {
                    exposure[index] += inv;
                    seen[index] = row;
                }
            }
        }
    }
    py::dict updated;
    for (size_t i = 0; i < keys.size(); ++i) {
        py::list pair(2);
        pair[0] = mass[i]; pair[1] = exposure[i];
        updated[py::int_(keys[i])] = std::move(pair);
    }
    return updated;
}

inline void bind(py::module_ &module) {
    module.def("commit_prefill_attention", &commit, py::arg("previous"), py::arg("decay"),
        py::arg("rows"), py::arg("starts"), py::arg("history"), py::arg("local_begin"), py::arg("masses"));
}
}  // namespace kvmem_prefill
