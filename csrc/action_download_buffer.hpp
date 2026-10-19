// Storage ownership for pooled pinned downloads. The legacy DLPack layouts
// below match the installed ATen/dlpack.h ABI; no Torch C++ library is linked.
#pragma once
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <cstdint>
#include <memory>
#include <vector>

namespace kvmem_download {
namespace py=pybind11;
struct Device {int32_t type,id;};
struct DType {uint8_t code,bits;uint16_t lanes;};
struct Tensor {void* data;Device device;int32_t ndim;DType dtype;int64_t *shape,*strides;uint64_t offset;};
struct Managed {Tensor tensor;void* context;void(*deleter)(Managed*);};
static_assert(sizeof(Tensor)==48 && sizeof(Managed)==64,"Unsupported legacy DLPack ABI");

struct Owner {
    Managed managed;
    std::vector<int64_t> shape;
    py::object backing,lease;
    static void release(Managed* tensor) {
        py::gil_scoped_acquire gil;
        delete static_cast<Owner*>(tensor->context);
    }
};

inline py::capsule capsule(uintptr_t pointer,size_t capacity,py::object backing,py::object lease,
        std::vector<int64_t> shape,int code,int bits) {
    if (!pointer || shape.empty() || shape.size()>8 || (code!=0 && code!=1 && code!=2) ||
            (bits!=8 && bits!=16 && bits!=32 && bits!=64))
        throw std::invalid_argument("Invalid pooled download storage geometry");
    size_t bytes=bits/8;
    for (const auto n:shape) {
        if (n<=0 || static_cast<uint64_t>(n)>capacity/bytes)
            throw std::invalid_argument("Pooled download shape exceeds reserved storage");
        bytes*=n;
    }
    auto state=std::make_unique<Owner>();state->shape=std::move(shape);
    state->backing=std::move(backing);state->lease=std::move(lease);
    state->managed=Managed{Tensor{reinterpret_cast<void*>(pointer),{1,0},
        static_cast<int32_t>(state->shape.size()),{static_cast<uint8_t>(code),static_cast<uint8_t>(bits),1},
        state->shape.data(),nullptr,0},state.get(),&Owner::release};
    py::capsule result(&state->managed,"dltensor",[](PyObject* value) {
        if (PyCapsule_IsValid(value,"dltensor")) {
            auto tensor=static_cast<Managed*>(PyCapsule_GetPointer(value,"dltensor"));
            tensor->deleter(tensor);
        }
    });
    state.release();return result;
}

inline void bind(py::module_& module) {module.def("leased_download_capsule",&capsule);}
}
