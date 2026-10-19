// Accepted output rows stay in their downloaded dense matrix. Validation and
// staging never mutate the published dictionary or advance the scope parser.
#pragma once
#include "block_graph_accept.hpp"

namespace kvmem_decode {
namespace py = pybind11;

inline py::tuple prepare(py::dict old, size_t samples, const std::vector<size_t>& indices,
        const std::vector<int64_t>& targets, py::buffer masses, size_t max_bytes) {
    const auto b = masses.request();
    const bool fp32 = b.format == py::format_descriptor<float>::format() && b.itemsize == 4;
    const bool fp64 = b.format == py::format_descriptor<double>::format() && b.itemsize == 8;
    if (b.ndim != 2 || (!fp32 && !fp64) || b.shape[1] != static_cast<ssize_t>(targets.size()))
        throw std::invalid_argument("Decode attention matrix shape/dtype mismatch");
    if (old.size() > max_bytes/320 || targets.size() > max_bytes/320 ||
            4096 + (old.size()+targets.size())*320 > max_bytes)
        throw std::length_error("Decode attention staging capacity exceeded before acceptance");
    std::unordered_map<int64_t,double> updated;
    for (auto item : old) updated.emplace(py::cast<int64_t>(item.first), py::cast<double>(item.second));
    std::vector<int64_t> additions;
    const auto read = fp32 ? &kvmem_graph::read<float> : &kvmem_graph::read<double>;
    {
        py::gil_scoped_release release;
        size_t previous = 0; bool first = true;
        for (auto row : indices) {
            if (row >= static_cast<size_t>(b.shape[0]) || (!first && row <= previous))
                throw std::invalid_argument("Invalid or duplicate decode accepted row index");
            first = false; previous = row;
            double sum = 0., correction = 0.;
            for (size_t col=0; col<targets.size(); ++col) {
                const double value = read(b,row,col);
                if (!std::isfinite(value) || value < 0.)
                    throw std::invalid_argument("Action attention mass is invalid or does not sum to one");
                const double next = sum+value;
#if PY_VERSION_HEX >= 0x030C0000
                correction += std::abs(sum) >= std::abs(value) ? (sum-next)+value : (value-next)+sum;
#endif
                sum = next;
            }
            sum += correction;
            if (!std::isfinite(sum) || std::abs(sum-1.) > .005)
                throw std::invalid_argument("Action attention mass is invalid or does not sum to one");
            for (size_t col=0; col<targets.size(); ++col) {
                const double value = read(b,row,col);
                // Preserve Python's sequential double additions, including
                // repeated fragments belonging to the same historical page.
                if (value != 0.) {
                    auto item = updated.emplace(targets[col],0.);
                    if (item.second) additions.push_back(targets[col]);
                    item.first->second += value;
                }
            }
        }
    }
    py::dict result = old.attr("copy")();
    for (auto key : additions) result[py::int_(key)] = py::float_(updated.at(key));
    for (const auto& item : updated) result[py::int_(item.first)] = py::float_(item.second);
    return py::make_tuple(result,samples+indices.size());
}

inline void bind(py::module_& module) {
    module.def("prepare_decode_attention",&prepare);
}
}
