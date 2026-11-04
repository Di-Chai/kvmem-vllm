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

    State committed(double decay,const std::vector<int64_t> &rows,
            const std::vector<int64_t> &starts,const std::vector<int64_t> &history,
            int64_t local_begin,py::buffer masses) const {
        const auto buffer=masses.request();
        const bool fp32=buffer.format==py::format_descriptor<float>::format() && buffer.itemsize==4;
        const bool fp64=buffer.format==py::format_descriptor<double>::format() && buffer.itemsize==8;
        if (buffer.ndim!=2 || (!fp32 && !fp64))
            throw std::invalid_argument("Prefill attention requires a float32/float64 matrix");
        if (buffer.shape[0]!=static_cast<ssize_t>(rows.size()))
            throw std::runtime_error("Missing prefill attention rows");
        if (buffer.shape[1]!=static_cast<ssize_t>(history.size()))
            throw std::runtime_error("Invalid prefill attention mass");
        if (rows.empty()) throw std::invalid_argument("Prefill attention requires sampled rows");
        State result;
        {
            py::gil_scoped_release release;
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
};
} // namespace

PYBIND11_MODULE(_prefill_state_native,module) {
    py::class_<State>(module,"State")
        .def_static("from_values",&State::from_values)
        .def("values",&State::values)
        .def("committed",&State::committed);
}
