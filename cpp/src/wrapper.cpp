#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "cones.h"
#include "deriv.h"
#include "linop.h"
#include "lsqr.h"

namespace py = pybind11;

PYBIND11_MODULE(_diffcp, m) {
  m.doc() = "Differentiating through Cone Programs C++ Extension";

  py::class_<LinearOperator>(m, "LinearOperator")
      .def("matvec", &LinearOperator::apply_matvec)
      .def("rmatvec", &LinearOperator::apply_rmatvec);
  py::class_<Cone>(m, "Cone")
      .def(py::init<ConeType, const std::vector<int> &>())
      .def_readonly("type", &Cone::type)
      .def_readonly("sizes", &Cone::sizes);
  py::enum_<ConeType>(m, "ConeType")
      .value("ZERO", ConeType::ZERO)
      .value("POS", ConeType::POS)
      .value("SOC", ConeType::SOC)
      .value("PSD", ConeType::PSD)
      .value("EXP", ConeType::EXP);
  m.def("_solve_derivative", &_solve_derivative);
  m.def("_solve_adjoint_derivative", &_solve_adjoint_derivative);
  m.def("dprojection", &dprojection);
  m.def("project_exp_cone", &project_exp_cone);
  m.def("in_exp", &in_exp);
  m.def("in_exp_dual", &in_exp_dual);
}
