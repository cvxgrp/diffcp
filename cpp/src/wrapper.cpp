#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

#include "deriv.h"
#include "lsqr.h"
#include "cones.h"

namespace py = pybind11;

PYBIND11_MODULE(_diffcp, m) {
  m.doc() = "Differentiating through Cone Programs C++ Extension";

  py::class_<LinearOperator>(m, "LinearOperator");
  py::class_<Cone>(m, "Cone")
    .def(py::init<ConeType, const std::vector<int> &>());
  py::enum_<ConeType>(m, "ConeType")
    .value("ZERO", ConeType::ZERO)
    .value("POS", ConeType::POS)
    .value("SOC", ConeType::SOC)
    .value("PSD", ConeType::PSD)
    .value("EXP", ConeType::EXP);
  m.def("_solve_derivative", &_solve_derivative);
  m.def("_solve_adjoint_derivative", &_solve_adjoint_derivative);
  m.def("dprojection", &dprojection);
}