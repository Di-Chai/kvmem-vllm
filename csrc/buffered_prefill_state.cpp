// Immutable native mass/exposure state for the existing prefill policy.
// Reuse the exact compensated validation sum, multiply and typed reads from
// the qualified dictionary backend. Only state representation changes.
#include "prefill_attention_commit.hpp"

namespace {
namespace py = pybind11;

struct State {
    std::vector<int64_t> keys;
    std::vector<double> mass, exposure;
    std::unordered_map<int64_t,size_t> positions;

    static State from_values(const py::dict &values) {
        State result;
        result.keys.reserve(values.size());result.mass.reserve(values.size());
        result.exposure.reserve(values.size());result.positions.reserve(values.size());
        for (const auto &item : values) {
            const auto key=py::cast<int64_t>(item.first);
            const auto pair=py::cast<std::vector<double>>(item.second);
            if (pair.size()!=2) throw std::invalid_argument("Invalid prefill history state");
            result.positions.emplace(key,result.keys.size());result.keys.push_back(key);
            result.mass.push_back(pair[0]);result.exposure.push_back(pair[1]);
        }
        return result;
    }

    py::dict values() const {
        py::dict result;
        for (size_t i=0;i<keys.size();++i) {
            py::list pair(2);pair[0]=mass[i];pair[1]=exposure[i];
            result[py::int_(keys[i])]=std::move(pair);
        }
        return result;
    }

    py::dict scores() const {
        // Selection only needs the ratio. Reuse the accepted native doubles
        // without materializing a Python [mass, exposure] list for every key.
        // Keep insertion order, filtering and IEEE division identical to
        // PrefillHistory.scores(); do not quantize or reassociate accumulation.
        py::dict result;
        for (size_t i=0;i<keys.size();++i) {
            if (exposure[i]>0.) result[py::int_(keys[i])]=mass[i]/exposure[i];
        }
        return result;
    }

    static bool validate_buffer(const py::buffer_info &buffer,
            const std::vector<int64_t> &rows,const std::vector<int64_t> &history) {
        const bool fp32=buffer.format==py::format_descriptor<float>::format() && buffer.itemsize==4;
        const bool fp64=buffer.format==py::format_descriptor<double>::format() && buffer.itemsize==8;
        if (buffer.ndim!=2 || (!fp32 && !fp64))
            throw std::invalid_argument("Prefill attention requires a float32/float64 matrix");
        if (buffer.shape[0]!=static_cast<ssize_t>(rows.size()))
            throw std::runtime_error("Missing prefill attention rows");
        if (buffer.shape[1]!=static_cast<ssize_t>(history.size()))
            throw std::runtime_error("Invalid prefill attention mass");
        if (rows.empty()) throw std::invalid_argument("Prefill attention requires sampled rows");
        return fp32;
    }

    // Called without the GIL by both entry points. Keep each layer's ordered
    // scalar arithmetic identical; batching only shares argument conversion.
    State committed_buffer(double decay,const std::vector<int64_t> &rows,
            const std::vector<int64_t> &starts,const std::vector<int64_t> &history,
            int64_t local_begin,const py::buffer_info &buffer,bool fp32) const {
        State result;
        {
            result.keys.reserve(history.size());result.mass.reserve(history.size());
            result.exposure.reserve(history.size());result.positions.reserve(history.size());
            std::vector<size_t> indices;indices.reserve(history.size());
            for (const auto key : history) {
                const auto inserted=result.positions.emplace(key,result.keys.size());
                indices.push_back(inserted.first->second);
                if (!inserted.second)continue;
                result.keys.push_back(key);
                const auto previous=positions.find(key);
                const bool found=previous!=positions.end();
                result.mass.push_back(kvmem_prefill::multiply(decay,found?mass[previous->second]:0.));
                result.exposure.push_back(kvmem_prefill::multiply(decay,found?exposure[previous->second]:0.));
            }
            const auto read_value=fp32?&kvmem_prefill::read<float>:&kvmem_prefill::read<double>;
            const double inv=1./rows.size();
            std::vector<ssize_t> seen(result.keys.size(),-1);
            for (ssize_t row=0;row<buffer.shape[0];++row) {
                kvmem_prefill::Sum sum;
                for (ssize_t page=0;page<buffer.shape[1];++page) {
                    const double value=read_value(buffer,row,page);
                    if (!std::isfinite(value) || value<0.)
                        throw std::runtime_error("Invalid prefill attention mass");
                    sum.add(value);
                }
                if (std::abs(sum.value()-1.)>.005)
                    throw std::runtime_error("Invalid prefill attention mass");
                if ((rows[row]>0 && local_begin>INT64_MAX-rows[row]) ||
                    (rows[row]<0 && local_begin<INT64_MIN-rows[row]))
                    throw std::invalid_argument("Prefill position overflow");
                const auto position=local_begin+rows[row];
                for (size_t page=0;page<std::min(starts.size(),history.size());++page) {
                    if (starts[page]>position)continue;
                    const auto index=indices[page];
                    result.mass[index]+=kvmem_prefill::multiply(inv,read_value(buffer,row,page));
                    if (seen[index]!=row) {
                        result.exposure[index]+=inv;seen[index]=row;
                    }
                }
            }
        }
        return result;
    }

    State committed(double decay,const std::vector<int64_t> &rows,
            const std::vector<int64_t> &starts,const std::vector<int64_t> &history,
            int64_t local_begin,py::buffer masses) const {
        const auto buffer=masses.request();
        const bool fp32=validate_buffer(buffer,rows,history);
        py::gil_scoped_release release;
        return committed_buffer(decay,rows,starts,history,local_begin,buffer,fp32);
    }
};

std::vector<State> committed_batch(py::iterable state_objects,
        const std::vector<double> &decays,const std::vector<int64_t> &rows,
        const std::vector<int64_t> &starts,
        const std::vector<std::vector<int64_t>> &histories,
        int64_t local_begin,py::buffer masses) {
    // Own a private immutable container before extracting pointers. The caller
    // may supply a generator or mutate its original list while the GIL is free.
    const py::tuple owners(state_objects);
    std::vector<const State *> states;
    states.reserve(owners.size());
    for (const auto owner : owners) states.push_back(py::cast<const State *>(owner));
    const auto buffer=masses.request();
    const bool fp32=buffer.format==py::format_descriptor<float>::format() && buffer.itemsize==4;
    const bool fp64=buffer.format==py::format_descriptor<double>::format() && buffer.itemsize==8;
    if (buffer.ndim!=3 || (!fp32 && !fp64))
        throw std::invalid_argument("Prefill attention requires a float32/float64 matrix");
    if (decays.size()!=states.size() || histories.size()!=states.size() ||
            buffer.shape[0]!=static_cast<ssize_t>(states.size()))
        throw std::invalid_argument("Missing independent prefill attention layers");
    for (const auto state : states)
        if (!state) throw std::invalid_argument("Missing native prefill attention state");

    // These descriptors borrow the original buffer, including signed strides.
    // The Python arguments and buffer lease stay alive until the GIL is held
    // again; no per-layer Tensor/numpy views or score copies are necessary.
    std::vector<py::buffer_info> layers;
    layers.reserve(states.size());
    for (size_t layer=0;layer<states.size();++layer)
        layers.emplace_back(static_cast<char *>(buffer.ptr)+static_cast<ssize_t>(layer)*buffer.strides[0],
            buffer.itemsize,buffer.format,2,
            std::vector<ssize_t>{buffer.shape[1],buffer.shape[2]},
            std::vector<ssize_t>{buffer.strides[1],buffer.strides[2]},buffer.readonly);
    std::vector<State> result;
    result.reserve(states.size());
    {
        py::gil_scoped_release release;
        for (size_t layer=0;layer<states.size();++layer) {
            // Validate and update in the same layer order as scalar commits.
            // Nothing is returned or published when any layer fails.
            const bool layer_fp32=State::validate_buffer(layers[layer],rows,histories[layer]);
            result.push_back(states[layer]->committed_buffer(decays[layer],rows,starts,
                histories[layer],local_begin,layers[layer],layer_fp32));
        }
    }
    return result;
}
} // namespace

PYBIND11_MODULE(_prefill_state_native,module) {
    py::class_<State>(module,"State")
        .def_static("from_values",&State::from_values)
        .def("values",&State::values)
        .def("scores",&State::scores)
        .def("committed",&State::committed);
    module.def("committed_batch",&committed_batch);
}
