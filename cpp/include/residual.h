#pragma once

#include "cones.h"
#include "eigen_includes.h"
#include "linop.h"
#include "lsqr.h"
#include "deriv.h"

std::tuple<Vector, Vector> proj_pq(const std::vector<Cone> &cones,
                                   const Vector &u, const Vector &v, double w);

/**
 * Refines the solution to a cone program by using LSQR to improve
 * the normalized residual map:
 * https://stanford.edu/~boyd/papers/pdf/cone_prog_refine.pdf
 */
Vector refine(const SparseMatrix &Q, const std::vector<Cone> &cones,
              const Vector &u, const Vector &v, double w,
              const int n_iter,
              const double lsqr_lambda=1e-8,
              const double lsqr_iter_lim=30,
              const int alpha_K=10);


/**
 * Compute the value of the residual map at a point z = (u, v, w)
 * for a cone program defined by a sparse Q.
 */
Vector residual(const SparseMatrix &Q, const std::vector<Cone> &cones,
                const Vector &u, const Vector &v, double w);

/**
 * Compute the value of the normalized residual map at a point z = (u, v, w)
 * for a cone program defined by a sparse Q.
 */
Vector N_residual(const SparseMatrix &Q, const std::vector<Cone> &cones,
                  const Vector &u, const Vector &v, double w);

/**
 * Compute the value of the residual map at a point z = (u, v, w)
 * for a cone program defined by a dense Q.
 */
Vector residual_dense(const Matrix &Q, const std::vector<Cone> &cones,
                      const Vector &u, const Vector &v, double w);

/**
 * Compute the value of the normalized residual map at a point z = (u, v, w)
 * for a cone program defined by a sparse Q.
 */
Vector N_residual_dense(const Matrix &Q, const std::vector<Cone> &cones,
                        const Vector &u, const Vector &v, double w);

/**
 * The derivative of the normalized residual map at a point z = (u, v, w)
 * for a cone program defined by a sparse Q.
 */
LinearOperator DN(const SparseMatrix &Q, const std::vector<Cone> &cones,
                  const Vector &u, const Vector &v, double w);

/**
 * The derivative of the normalized residual map at a point z = (u, v, w)
 * for a cone program defined by a dense Q.
 */
Matrix DN_dense(const Matrix &Q, const std::vector<Cone> &cones, const Vector &u,
                const Vector &v, double w);

/**
 * Left-multiply a point dz = (du, dv, dw) by the derivative of the
 * normalized residual map.
 */
Vector apply_DN_matvec(const SparseMatrix &Q, const std::vector<Cone> &cones,
                       const Vector &u, const Vector &v, double w,
                       const Vector &du, const Vector &dv, double dw);

/**
 * Right-multiply a point dz = (du, dv, dw) by the derivative of the
 * normalized residual map.
 */
Vector apply_DN_rmatvec(const SparseMatrix &Q, const std::vector<Cone> &cones,
                        const Vector &u, const Vector &v, double w,
                        const Vector &du, const Vector &dv, double dw);
