#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

#include "deriv.h"
#include "lsqr.h"
#include "cones.h"

namespace py = pybind11;

PYBIND11_MODULE(_diffcp, m) {
  m.doc() = "Differentiating through Cone Programs C++ Extension";

  m.def("_solve_derivative", &_solve_derivative);
  m.def("_solve_adjoint_derivative", &_solve_adjoint_derivative);
}