#include "residual.h"
#include "lsqr.h"
#include "deriv.h"
#include <tuple>
#include <iostream>

inline double gt(double x, double t) {
  if (x >= t) {
    return 1.0;
  } else {
    return 0.0;
  }
}

std::tuple<Vector, Vector> proj_pq(const std::vector<Cone> &cones,
          const Vector &u, const Vector &v, double w) {
  int n = u.size();
  int m = v.size();
  int N = n + m + 1;
  Vector z = Vector::Zero(N);
  z << u, v, w;
  Vector p = Vector(N);
  p << u, projection(v, cones, true), std::max(w, 0.0);
  Vector q = p - z;
  return {p, q};
}

Vector residual(const SparseMatrix &Q, const std::vector<Cone> &cones,
                const Vector &u, const Vector &v, double w) {
  auto [p, q] = proj_pq(cones, u, v, w);
  return Q*p - q;
}

Vector N_residual(const SparseMatrix &Q, const std::vector<Cone> &cones,
                  const Vector &u, const Vector &v, double w) {
  return residual(Q, cones, u, v, w) / w;
}

Vector residual_dense(const Matrix &Q, const std::vector<Cone> &cones,
                      const Vector &u, const Vector &v, double w) {
  auto [p, q] = proj_pq(cones, u, v, w);
  return Q*p - q;
}

Vector N_residual_dense(const Matrix &Q, const std::vector<Cone> &cones,
                        const Vector &u, const Vector &v, double w) {
  return residual_dense(Q, cones, u, v, w) / w;
}


Vector apply_DN_matvec(const SparseMatrix &Q, const std::vector<Cone> &cones,
                       const Vector &u, const Vector &v, double w,
                       const Vector &du, const Vector &dv, double dw) {
  int N = u.size() + v.size() + 1;
  LinearOperator dpi_op = dpi(u, v, w, cones);

  Vector dz = Vector::Zero(N);
  dz << du, dv, dw;

  Vector dpi_mv = dpi_op.matvec(dz);
  Vector result = (Q*dpi_mv - dpi_mv + dz) / abs(w);
  float dz_w = dz[N-1];
  Vector last_row_offset = gt(w, 0.0) * residual(Q, cones, u, v, w) / (w*w);
  result -= dz_w * last_row_offset;
  return result;
}

Vector apply_DN_rmatvec(const SparseMatrix &Q, const std::vector<Cone> &cones,
                        const Vector &u, const Vector &v, double w,
                        const Vector &du, const Vector &dv, double dw) {
  int N = u.size() + v.size() + 1;
  LinearOperator dpi_op = dpi(u, v, w, cones);

  Vector dz = Vector::Zero(N);
  dz << du, dv, dw;

  Vector t = Q.transpose() * dz - dz;
  Vector dpi_t_rmv = dpi_op.rmatvec(t);
  Vector result = (dpi_t_rmv + dz) / abs(w);
  Vector last_row_offset = gt(w, 0.0) * residual(Q, cones, u, v, w) / (w*w);
  result[N-1] -= last_row_offset.dot(dz);
  return result;
}


LinearOperator DN(const SparseMatrix &Q, const std::vector<Cone> &cones,
                  const Vector &u, const Vector &v, double w) {
  int N = u.size() + v.size() + 1;
  LinearOperator dpi_op = dpi(u, v, w, cones);

  // TODO: Potentially remove duplicated code applying the matvec/rmatvec,
  // but keeping for now to cache dpi_op.
  const VecFn matvec = [Q,cones,u,v,w,N,dpi_op](const Vector &dz) -> Vector {
    Vector dpi_mv = dpi_op.matvec(dz);
    Vector result = (Q*dpi_mv - dpi_mv + dz) / abs(w);
    float dz_w = dz[N-1];
    Vector last_row_offset = gt(w, 0.0) * residual(Q, cones, u, v, w) / (w*w);
    result -= dz_w * last_row_offset;
    return result;
  };
  const VecFn rmatvec = [Q,cones,u,v,w,N,dpi_op](const Vector &dz) -> Vector {
    Vector t = Q.transpose() * dz - dz;
    Vector dpi_t_rmv = dpi_op.rmatvec(t);
    Vector result = (dpi_t_rmv + dz) / abs(w);
    Vector last_row_offset = gt(w, 0.0) * residual(Q, cones, u, v, w) / (w*w);
    result[N-1] -= last_row_offset.dot(dz);
    return result;
  };
  return LinearOperator(N, N, matvec, rmatvec);
}

Matrix DN_dense(const Matrix &Q, const std::vector<Cone> &cones,
                const Vector &u, const Vector &v, double w) {
  int n = u.size();
  int m = v.size();
  int N = n + m + 1;
  Matrix eye = Matrix::Identity(N, N);
  Matrix DN = (Q - eye) * dpi_dense(u, v, w, cones) + eye;
  DN /= abs(w);

  Vector last_row_offset = gt(w, 0.0) * residual_dense(Q, cones, u, v, w) / (w*w);
  DN.col(N-1) -= last_row_offset;
  return DN;
}

Vector refine(const SparseMatrix &Q, const std::vector<Cone> &cones,
              const Vector &in_u, const Vector &in_v, double in_w,
              const int n_iter,
              const double lambda,
              const double lsqr_iter_lim,
              const int alpha_K) {
  int n = in_u.size();
  int m = in_v.size();
  int N = n + m + 1;

  Vector z = Vector(N);
  z << in_u, in_v, in_w;

  for (int i = 0; i < n_iter; ++i) {
    Vector z_u = z.segment(0, n);
    Vector z_v = z.segment(n, m);
    double z_w = z[N-1];
    Vector res = N_residual(Q, cones, z_u, z_v, z_w);
    double res_norm = res.norm();

    LinearOperator DN_op = DN(Q, cones, z_u, z_v, z_w);
    double lsqr_atol = 1e-8;
    double lsqr_btol = 1e-8;
    LsqrResult result = lsqr(DN_op, res, lambda, lsqr_atol, lsqr_btol,
                             1e8, lsqr_iter_lim);
    Vector step = result.x;

    int alpha_k = 0;
    double next_norm = res_norm + 1.;
    Vector next_z;
    while ((next_norm > res_norm) & (alpha_k < alpha_K)) {
      double alpha = 1./((double) (2 << alpha_k));
      next_z = z - alpha * step;
      Vector next_z_u = next_z.segment(0, n);
      Vector next_z_v = next_z.segment(n, m);
      double next_z_w = next_z[N-1];
      Vector next_res = N_residual(Q, cones, next_z_u, next_z_v, next_z_w);
      next_norm = next_res.norm();
      alpha_k += 1;
    }
    z << next_z;
  }

  return z;
}
