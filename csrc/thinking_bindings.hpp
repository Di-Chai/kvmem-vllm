#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "historical_thinking.hpp"

inline void bind_historical_thinking(pybind11::module_ &m) {
    namespace py = pybind11;
    using Parser = vllm_kvmem::HistoricalThinkingParser;
    py::class_<Parser>(m, "HistoricalThinkingParser")
        .def(py::init<std::vector<uint32_t>, std::vector<std::vector<uint32_t>>>(),
             py::arg("markers"), py::arg("headers"))
        .def("feed", &Parser::feed)
        .def("seed", &Parser::seed)
        .def("reset", &Parser::reset)
        .def_property_readonly("thinking", &Parser::thinking);
}
