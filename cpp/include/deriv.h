#pragma once

#include "cones.h"
#include "eigen_includes.h"
#include "linop.h"

LinearOperator M_operator(const SparseMatrix &Q, const std::vector<Cone> &cones,
                          const Vector &u, const Vector &v, double w);

Matrix M_dense(const Matrix& Q, const std::vector<Cone>& cones, const Vector& u, const Vector& v, double w);
Vector _solve_derivative_dense(const Matrix& M, const Vector& rhs);
Vector _solve_adjoint_derivative_dense(const Matrix& MT, const Vector& dz);
