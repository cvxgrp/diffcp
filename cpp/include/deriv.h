#pragma once

#include "cones.h"
#include "eigen_includes.h"
#include "linop.h"
#include "lsqr.h"

LinearOperator dpi(const Vector &u, const Vector &v, double w,
                   const std::vector<Cone> &cones);

Matrix dpi_dense(const Vector &u, const Vector &v, double w,
                 const std::vector<Cone> &cones);

LinearOperator M_operator(const SparseMatrix &Q, const std::vector<Cone> &cones,
                          const Vector &u, const Vector &v, double w);

Matrix M_dense(const Matrix &Q, const std::vector<Cone> &cones, const Vector &u,
               const Vector &v, double w);

LsqrResult _solve_adjoint_derivative_lsqr(
    const SparseMatrix &Q, const std::vector<Cone> &cones,
    const Vector &u, const Vector &v, double w, const Vector &dz);


// this function releases the GIL.
Vector _solve_derivative_dense(const Matrix &M, const Matrix &MT,
                               const Vector &rhs);

// this function releases the GIL.
Vector _solve_adjoint_derivative_dense(const Matrix &M, const Matrix &MT,
                                       const Vector &dz);
