// Dense accepted attention feedback, without Python scalar loops. The caller
// reserves staging and holds the graph lock; this function never publishes it.
#pragma once
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <numeric>
#include <unordered_map>
#include <vector>

namespace kvmem_graph {
namespace py = pybind11;

// Expansion sum with the final half-even fix used by math.fsum. Inputs are
// finite nonnegative doubles, so NaN/infinity cancellation is not applicable.
struct ExactSum {
    std::vector<double> partials;
    void add(double x) {
        size_t i = 0;
        for (double y : partials) {
            if (std::abs(x) < std::abs(y)) std::swap(x, y);
            const double hi = x+y;
            const double lo = y-(hi-x);
            if (lo != 0.) partials[i++] = lo;
            x = hi;
        }
        partials.resize(i);
        partials.push_back(x);
    }
    double value() const {
        size_t n = partials.size();
        if (!n) return 0.;
        double hi = partials[--n], lo = 0.;
        while (n) {
            const double x = hi, y = partials[--n];
            hi = x+y;
            lo = y-(hi-x);
            if (lo != 0.) break;
        }
        if (n && ((lo < 0. && partials[n-1] < 0.) || (lo > 0. && partials[n-1] > 0.))) {
            const double y = lo*2., x = hi+y;
            if (y == x-hi) hi = x;
        }
        return hi;
    }
};

struct Row {
    int64_t source;
    std::vector<int64_t> targets;
    std::vector<double> values;
    int64_t samples = 0, prefill = 0, output = 0;
};

template<class T>
double read(const py::buffer_info &b, ssize_t row, ssize_t column) {
    T value;
    std::memcpy(&value, static_cast<const char *>(b.ptr)+row*b.strides[0]+column*b.strides[1], sizeof(T));
    return static_cast<double>(value);
}

inline py::tuple prepare(const std::vector<int64_t> &positions,
        const std::vector<int64_t> &targets, py::buffer masses,
        int64_t accepted_through, int64_t end, int64_t block, size_t topk,
        double min_mass, bool prefill, py::object old_partial,
        py::object selected = py::none()) {
    const auto buffer = masses.request();
    const bool fp32 = buffer.format == py::format_descriptor<float>::format() && buffer.itemsize == 4;
    const bool fp64 = buffer.format == py::format_descriptor<double>::format() && buffer.itemsize == 8;
    std::vector<size_t> accepted_rows;
    if (!selected.is_none()) accepted_rows = selected.cast<std::vector<size_t>>();
    if (buffer.ndim != 2 || (!fp32 && !fp64) ||
            (selected.is_none() ? buffer.shape[0] != static_cast<ssize_t>(positions.size()) :
                accepted_rows.size() != positions.size()) ||
            buffer.shape[1] != static_cast<ssize_t>(targets.size()))
        throw std::invalid_argument("Graph attention matrix shape/dtype mismatch");
    for (const auto index : accepted_rows)
        if (index >= static_cast<size_t>(buffer.shape[0]))
            throw std::invalid_argument("Graph accepted row index is out of bounds");
    if (block <= 0 || topk == 0 || accepted_through < 0 || end < accepted_through ||
            !std::isfinite(min_mass) || min_mass < 0. || (!positions.empty() && targets.empty()))
        throw std::invalid_argument("Invalid dense graph geometry");
    for (const auto target : targets)
        if (target < 0 || target % block) throw std::invalid_argument("Invalid graph target");
    for (const auto position : positions)
        if (position < 0 || position >= end) throw std::invalid_argument("Invalid graph source position");

    Row pending{-1, {}, {}};
    if (!old_partial.is_none()) {
        pending.source = old_partial.attr("source").cast<int64_t>();
        pending.targets = old_partial.attr("targets").cast<std::vector<int64_t>>();
        pending.values = old_partial.attr("sums").cast<std::vector<double>>();
        pending.samples = old_partial.attr("samples").cast<int64_t>();
        pending.prefill = old_partial.attr("prefill_samples").cast<int64_t>();
        pending.output = old_partial.attr("output_samples").cast<int64_t>();
        if (pending.source < 0 || pending.source % block || pending.samples <= 0 ||
                pending.targets.size() != pending.values.size() || pending.prefill < 0 || pending.output < 0)
            throw std::invalid_argument("Invalid graph partial state");
        for (size_t i = 0; i < pending.targets.size(); ++i)
            if (pending.targets[i] < 0 || pending.targets[i] % block ||
                    !std::isfinite(pending.values[i]) || pending.values[i] < 0.)
                throw std::invalid_argument("Invalid graph partial mass/target");
    }
    std::vector<Row> completed;
    const auto read_value = fp32 ? &read<float> : &read<double>;
    {
        py::gil_scoped_release release;
        // One shared key index per matrix, including any inherited partial.
        std::vector<int64_t> keys;
        std::vector<size_t> indices;
        std::unordered_map<int64_t,size_t> lookup;
        auto index = [&](int64_t key) {
            const auto inserted = lookup.emplace(key, keys.size());
            if (inserted.second) keys.push_back(key);
            return inserted.first->second;
        };
        for (auto key : pending.targets) index(key);
        for (auto key : targets) indices.push_back(index(key));
        std::vector<double> totals(keys.size(),0.), values(keys.size(),0.);
        std::vector<bool> touched(keys.size(),false);
        for (size_t i = 0; i < pending.targets.size(); ++i) {
            totals[lookup.at(pending.targets[i])] = pending.values[i];
            touched[lookup.at(pending.targets[i])] = true;
        }
        auto finish = [&]() {
            if (pending.samples) {
                std::vector<std::pair<int64_t,double>> candidates;
                for (size_t i = 0; i < keys.size(); ++i) {
                    const double weight = totals[i]/pending.samples;
                    if (keys[i] < pending.source && totals[i] > 0. && weight >= min_mass)
                        candidates.emplace_back(keys[i],weight);
                }
                const size_t keep = std::min(topk,candidates.size());
                std::partial_sort(candidates.begin(),candidates.begin()+keep,candidates.end(),
                    [](const auto &a, const auto &b) {
                        return a.second != b.second ? a.second > b.second : a.first < b.first;
                    });
                Row sealed{pending.source, {}, {}, pending.samples,pending.prefill,pending.output};
                for (size_t i = 0; i < keep; ++i) {
                    sealed.targets.push_back(candidates[i].first);
                    sealed.values.push_back(candidates[i].second);
                }
                completed.push_back(std::move(sealed));
            }
            pending = Row{-1, {}, {}};
            std::fill(totals.begin(),totals.end(),0.);
            std::fill(touched.begin(),touched.end(),false);
        };
        std::vector<size_t> order(positions.size());
        std::iota(order.begin(),order.end(),0);
        std::sort(order.begin(),order.end(),[&](size_t a,size_t b){return positions[a]<positions[b];});
        for (size_t i = 1; i < order.size(); ++i)
            if (positions[order[i-1]] == positions[order[i]])
                throw std::invalid_argument("Duplicate graph source position");
        for (size_t row : order) {
            ExactSum sum;
            for (size_t col = 0; col < targets.size(); ++col) {
                const double v = read_value(buffer,selected.is_none() ? row : accepted_rows[row],col);
                if (!std::isfinite(v) || v < 0.) throw std::invalid_argument("Invalid graph attention mass");
                sum.add(v);
                if (positions[row] >= accepted_through && targets[col] > positions[row] && v > 0.)
                    throw std::invalid_argument("Graph sampled mass points to a future token block");
            }
            const double total = sum.value();
            if (!std::isfinite(total) || std::abs(total-1.) > .005)
                throw std::invalid_argument("Graph attention row does not sum to one");
            if (positions[row] < accepted_through) continue;
            const auto source = positions[row]/block*block;
            if (pending.source >= 0 && source != pending.source) finish();
            pending.source = source;
            std::fill(values.begin(),values.end(),0.);
            for (size_t col = 0; col < targets.size(); ++col) {
                const double v = read_value(buffer,selected.is_none() ? row : accepted_rows[row],col);
                if (v != 0.) {
                    // Keep duplicate-segment accumulation and per-sample
                    // normalization in their original double operation order.
                    values[indices[col]] += v/total;
                    touched[indices[col]] = true;
                }
            }
            for (size_t i = 0; i < keys.size(); ++i) totals[i] += values[i];
            ++pending.samples;
            if (prefill) ++pending.prefill; else ++pending.output;
        }
        if (pending.source >= 0 && pending.source <= end-block) finish();
        if (pending.source >= 0) {
            pending.targets.clear(); pending.values.clear();
            std::vector<size_t> order_keys(keys.size());
            std::iota(order_keys.begin(),order_keys.end(),0);
            std::sort(order_keys.begin(),order_keys.end(),[&](size_t a,size_t b){return keys[a]<keys[b];});
            for (auto i : order_keys) if (touched[i]) {
                pending.targets.push_back(keys[i]); pending.values.push_back(totals[i]);
            }
        }
    }
    const auto pack = [](const Row &row) {
        return py::make_tuple(row.source,py::tuple(py::cast(row.targets)),py::tuple(py::cast(row.values)),
                              row.samples,row.prefill,row.output);
    };
    py::list sealed;
    for (const auto &row : completed) sealed.append(pack(row));
    return py::make_tuple(sealed,pending.source < 0 ? py::none() : py::object(pack(pending)));
}

inline void bind(py::module_ &module) {
    module.def("prepare_block_graph",&prepare, py::arg("positions"), py::arg("targets"),
        py::arg("masses"), py::arg("accepted_through"), py::arg("end"), py::arg("block"),
        py::arg("topk"), py::arg("min_mass"), py::arg("prefill"), py::arg("old_partial"),
        py::arg("row_indices") = py::none());
}
}  // namespace kvmem_graph
