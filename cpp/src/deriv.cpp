#include "deriv.h"
#include "lsqr.h"

inline double gt(double x, double t) {
  if (x >= t) {
    return 1.0;
  } else {
    return 0.0;
  }
}

LinearOperator dpi(const Vector &u, const Vector &v, double w,
                   const std::vector<Cone> &cones) {
  LinearOperator eye = identity(u.size());
  LinearOperator D_proj = dprojection(v, cones, true);
  LinearOperator last = scalar(gt(w, 0.0));

  std::vector<LinearOperator> linops{eye, D_proj, last};

  return block_diag(linops);
}

LinearOperator M_operator(const SparseMatrix &Q, const std::vector<Cone> &cones,
                          const Vector &u, const Vector &v, double w) {
  int N = u.size() + v.size() + 1;
  return (aslinearoperator(Q) - identity(N)) * dpi(u, v, w, cones) +
         identity(N);
}
