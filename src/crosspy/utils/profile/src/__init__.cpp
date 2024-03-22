#include "Timer.h"

#include <pybind11/pybind11.h>
namespace py = pybind11;

PYBIND11_MODULE(profile, m) {
    py::class_<TimeAccumulator>(m, "Timer")
        .def(py::init())
        .def("start", &TimeAccumulator::start)
        .def("stop", &TimeAccumulator::stop)
        .def("get", &TimeAccumulator::get);
}