#pragma once

#include "eigen_includes.h"
#include "linop.h"
#include "cones.h"

Vector _solve_derivative(const SparseMatrix& Q, const std::vector<Cone>& cones,
    const Vector& u, const Vector& v, double w, const Vector& rhs);
Vector _solve_adjoint_derivative(const SparseMatrix& Q, const std::vector<Cone>& cones,
    const Vector& u, const Vector& v, double w, const Vector& dz);