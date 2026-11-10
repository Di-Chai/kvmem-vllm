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

// KVMEM_NATIVE_DECODE_SPARSE_MASS: prepare() without the final rewrite of every
// unchanged entry. Same validation, same sequential additions, same result
// contents and insertion order: only keys whose value changed, new keys and old
// non-float values (prepare() converts those to float) are written after copy().
inline py::tuple prepare_sparse(py::dict old, size_t samples, const std::vector<size_t>& indices,
        const std::vector<int64_t>& targets, py::buffer masses, size_t max_bytes) {
    const auto b = masses.request();
    const bool fp32 = b.format == py::format_descriptor<float>::format() && b.itemsize == 4;
    const bool fp64 = b.format == py::format_descriptor<double>::format() && b.itemsize == 8;
    if (b.ndim != 2 || (!fp32 && !fp64) || b.shape[1] != static_cast<ssize_t>(targets.size()))
        throw std::invalid_argument("Decode attention matrix shape/dtype mismatch");
    if (old.size() > max_bytes/320 || targets.size() > max_bytes/320 ||
            4096 + (old.size()+targets.size())*320 > max_bytes)
        throw std::length_error("Decode attention staging capacity exceeded before acceptance");
    // value, state: 0 unchanged old entry, 1 changed old entry, 2 new key.
    std::unordered_map<int64_t,std::pair<double,char>> updated;
    std::vector<int64_t> rewrite;
    updated.reserve(old.size() + targets.size());
    for (auto item : old) {
        const int64_t key = py::cast<int64_t>(item.first);
        updated.emplace(key, std::make_pair(py::cast<double>(item.second), char{0}));
        if (!PyFloat_CheckExact(item.second.ptr())) rewrite.push_back(key);
    }
    std::vector<int64_t> additions, touched;
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
                if (value != 0.) {
                    auto item = updated.emplace(targets[col], std::make_pair(0., char{2}));
                    if (item.second) additions.push_back(targets[col]);
                    else if (item.first->second.second == 0) {
                        item.first->second.second = 1;
                        touched.push_back(targets[col]);
                    }
                    item.first->second.first += value;
                }
            }
        }
    }
    py::dict result = old.attr("copy")();
    for (auto key : additions) result[py::int_(key)] = py::float_(updated.at(key).first);
    for (auto key : touched) result[py::int_(key)] = py::float_(updated.at(key).first);
    for (auto key : rewrite) result[py::int_(key)] = py::float_(updated.at(key).first);
    return py::make_tuple(result,samples+indices.size());
}

// KVMEM_NATIVE_DECODE_MASS_MAP: the accepted-mass dictionary kept natively.
// keys/values are in Python dict insertion order; index is (key, position)
// sorted by key. Copying is three flat vectors, so staging a prepare no longer
// rebuilds a Python dict of every attended block.
struct MassMap {
    std::vector<int64_t> keys;
    std::vector<double> values;
    std::vector<std::pair<int64_t,uint32_t>> index;

    size_t size() const { return keys.size(); }
    int64_t find(int64_t key) const {
        auto it = std::lower_bound(index.begin(), index.end(), std::make_pair(key, uint32_t{0}),
            [](const auto& a, const auto& b) { return a.first < b.first; });
        return it != index.end() && it->first == key ? static_cast<int64_t>(it->second) : -1;
    }
    static MassMap from_dict(const py::dict& old) {
        MassMap out;
        out.keys.reserve(old.size()); out.values.reserve(old.size()); out.index.reserve(old.size());
        for (auto item : old) {
            // The same casts, in the same order, as prepare().
            const int64_t key = py::cast<int64_t>(item.first);
            const double value = py::cast<double>(item.second);
            out.index.emplace_back(key, static_cast<uint32_t>(out.keys.size()));
            out.keys.push_back(key);
            out.values.push_back(value);
        }
        std::sort(out.index.begin(), out.index.end());
        return out;
    }
    py::dict to_dict() const {
        py::dict out;
        for (size_t i = 0; i < keys.size(); ++i) {
            py::object k = py::reinterpret_steal<py::object>(PyLong_FromLongLong(keys[i]));
            py::object v = py::reinterpret_steal<py::object>(PyFloat_FromDouble(values[i]));
            if (!k || !v || PyDict_SetItem(out.ptr(), k.ptr(), v.ptr()) < 0) throw py::error_already_set();
        }
        return out;
    }
};

// prepare() over a MassMap (or a dict, converted at the point prepare() casts
// it). Same checks in the same order, the same sequential double additions,
// and the same resulting contents and insertion order as prepare()'s dict.
inline py::tuple prepare_map(py::object old, size_t samples, const std::vector<size_t>& indices,
        const std::vector<int64_t>& targets, py::buffer masses, size_t max_bytes) {
    const auto b = masses.request();
    const bool fp32 = b.format == py::format_descriptor<float>::format() && b.itemsize == 4;
    const bool fp64 = b.format == py::format_descriptor<double>::format() && b.itemsize == 8;
    if (b.ndim != 2 || (!fp32 && !fp64) || b.shape[1] != static_cast<ssize_t>(targets.size()))
        throw std::invalid_argument("Decode attention matrix shape/dtype mismatch");
    const bool native = py::isinstance<MassMap>(old);
    if (!native && !py::isinstance<py::dict>(old))
        throw py::type_error("Decode attention mass must be a dict or DecodeMassMap");
    const size_t old_size = native ? old.cast<const MassMap&>().size() : py::len(old);
    if (old_size > max_bytes/320 || targets.size() > max_bytes/320 ||
            4096 + (old_size+targets.size())*320 > max_bytes)
        throw std::length_error("Decode attention staging capacity exceeded before acceptance");
    MassMap result = native ? old.cast<const MassMap&>() : MassMap::from_dict(old.cast<py::dict>());
    std::vector<std::pair<int64_t,uint32_t>> added;
    std::unordered_map<int64_t,uint32_t> added_at;
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
                if (value == 0.) continue;
                const int64_t key = targets[col];
                int64_t pos = result.find(key);
                if (pos < 0) {
                    auto hit = added_at.find(key);
                    if (hit != added_at.end()) pos = hit->second;
                }
                if (pos < 0) {
                    // prepare(): emplace(key, 0.) then += value; first touch order.
                    pos = static_cast<int64_t>(result.keys.size());
                    result.keys.push_back(key);
                    result.values.push_back(0.);
                    added_at.emplace(key, static_cast<uint32_t>(pos));
                    added.emplace_back(key, static_cast<uint32_t>(pos));
                }
                result.values[static_cast<size_t>(pos)] += value;
            }
        }
        if (!added.empty()) {
            std::sort(added.begin(), added.end());
            std::vector<std::pair<int64_t,uint32_t>> merged;
            merged.reserve(result.index.size() + added.size());
            std::merge(result.index.begin(), result.index.end(), added.begin(), added.end(),
                std::back_inserter(merged));
            result.index.swap(merged);
        }
    }
    return py::make_tuple(py::cast(std::move(result)), samples+indices.size());
}

inline void bind(py::module_& module) {
    module.def("prepare_decode_attention",&prepare);
    module.def("prepare_decode_attention_sparse",&prepare_sparse);
    py::class_<MassMap>(module, "DecodeMassMap")
        .def_static("from_dict", &MassMap::from_dict)
        .def("to_dict", &MassMap::to_dict)
        .def("__len__", &MassMap::size);
    module.def("prepare_decode_attention_map", &prepare_map);
}
}
