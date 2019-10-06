#pragma once

#include "eigen_includes.h"
#include "linop.h"
#include "cones.h"

Vector _solve_derivative(const SparseMatrix& Q, const std::vector<Cone>& cones,
    const Vector& u, const Vector& v, double w, const Vector& rhs);
Vector _solve_adjoint_derivative(const SparseMatrix& Q, const std::vector<Cone>& cones,
    const Vector& u, const Vector& v, double w, const Vector& dz);

Matrix M_dense(const Matrix& Q, const std::vector<Cone>& cones,
    const Vector& u, const Vector& v, double w);
Vector _solve_derivative_dense(const Matrix& M, const Vector& rhs);
Vector _solve_adjoint_derivative_dense(const Matrix& MT, const Vector& dz);

// TODO: These should also have (at least) rho/it_ref_iters and MT_iref
SparseMatrix M_sparse(const SparseMatrix& Q, const std::vector<Cone>& cones,
    const Vector& u, const Vector& v, double w);
            rho = kwargs.get("rho", 1e-6)
            it_ref_iters = kwargs.get("it_ref_iters", 5)
Vector _solve_derivative_sparse(const SparseMatrix& M, const Vector& rhs);
Vector _solve_adjoint_derivative_sparse(const SparseMatrix& MT, const Vector& dz);
