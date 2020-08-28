#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "cones.h"
#include "deriv.h"
#include "linop.h"
#include "lsqr.h"
#include "residual.h"

namespace py = pybind11;

PYBIND11_MODULE(_diffcp, m) {
  m.doc() = "Differentiating through Cone Programs C++ Extension";

  py::class_<LinearOperator>(m, "LinearOperator")
      .def("matvec", &LinearOperator::apply_matvec)
      .def("rmatvec", &LinearOperator::apply_rmatvec)
      .def("transpose", &LinearOperator::transpose);
  py::class_<Cone>(m, "Cone")
      .def(py::init<ConeType, const std::vector<int> &>())
      .def_readonly("type", &Cone::type)
      .def_readonly("sizes", &Cone::sizes);
  py::enum_<ConeType>(m, "ConeType")
      .value("ZERO", ConeType::ZERO)
      .value("POS", ConeType::POS)
      .value("SOC", ConeType::SOC)
      .value("PSD", ConeType::PSD)
      .value("EXP", ConeType::EXP)
      .value("EXP_DUAL", ConeType::EXP_DUAL);
  py::class_<LsqrResult>(m, "LsqrResult")
      .def_readonly("solution", &LsqrResult::x)
      .def_readonly("istop", &LsqrResult::istop)
      .def_readonly("itn", &LsqrResult::itn);
  m.def("lsqr_sparse", &lsqr_sparse,
        "Computes least-squares solution to sparse linear system via LSQR",
        py::arg("A"), py::arg("rhs"), py::arg("damp") = 0.0,
        py::arg("atol") = 1e-8, py::arg("btol") = 1e-8, py::arg("conlim") = 1e8,
        py::arg("iter_lim") = -1, py::call_guard<py::gil_scoped_release>());
  m.def("lsqr", &lsqr,
        "Computes least-squares solution to abstract linear system via LSQR",
        py::arg("A"), py::arg("rhs"), py::arg("damp") = 0.0,
        py::arg("atol") = 1e-8, py::arg("btol") = 1e-8, py::arg("conlim") = 1e8,
        py::arg("iter_lim") = -1, py::call_guard<py::gil_scoped_release>());

  m.def("M_operator", &M_operator, py::call_guard<py::gil_scoped_release>());
  m.def("M_dense", &M_dense, py::call_guard<py::gil_scoped_release>());
  m.def("_solve_derivative_dense", &_solve_derivative_dense, py::call_guard<py::gil_scoped_release>());
  m.def("_solve_adjoint_derivative_dense", &_solve_adjoint_derivative_dense, py::call_guard<py::gil_scoped_release>());
  m.def("_solve_adjoint_derivative_lsqr", &_solve_adjoint_derivative_lsqr, py::call_guard<py::gil_scoped_release>());

  m.def("projection", &projection, py::call_guard<py::gil_scoped_release>());
  m.def("dprojection", &dprojection, py::call_guard<py::gil_scoped_release>());
  m.def("dprojection_dense", &dprojection_dense, py::call_guard<py::gil_scoped_release>());
  m.def("dpi", &dpi, py::call_guard<py::gil_scoped_release>());
  m.def("dpi_dense", &dpi_dense, py::call_guard<py::gil_scoped_release>());
  m.def("project_exp_cone", &project_exp_cone);
  m.def("in_exp", &in_exp);
  m.def("in_exp_dual", &in_exp_dual);

  // residual.h
  m.def("proj_pq", &proj_pq, py::call_guard<py::gil_scoped_release>());
  m.def("residual", &residual, py::call_guard<py::gil_scoped_release>());
  m.def("N_residual", &N_residual, py::call_guard<py::gil_scoped_release>());
  m.def("residual_dense", &residual_dense, py::call_guard<py::gil_scoped_release>());
  m.def("N_residual_dense", &N_residual_dense, py::call_guard<py::gil_scoped_release>());
  m.def("DN", &DN, py::call_guard<py::gil_scoped_release>());
  m.def("apply_DN_matvec", &apply_DN_matvec, py::call_guard<py::gil_scoped_release>());
  m.def("apply_DN_rmatvec", &apply_DN_rmatvec, py::call_guard<py::gil_scoped_release>());
  m.def("DN_dense", &DN_dense, py::call_guard<py::gil_scoped_release>());
  m.def("refine", &refine, py::call_guard<py::gil_scoped_release>());
}
