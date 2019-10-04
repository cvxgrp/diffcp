#include "linop.h"
#include "lsqr.h"
#include <iostream>
#include <vector>

int main() {
  const VecFn matvec =
      [](const Vector &x) -> Vector {
    Vector y = Vector::Zero(2);
    y[0] = 1 * x[0] + 2 * x[1];
    y[1] = 3 * x[0] + 4 * x[1];
    return y;
  };

  const VecFn rmatvec = [](const Vector &x) -> Vector {
    Vector y = Vector::Zero(2);
    y[0] = 1 * x[0] + 3 * x[1];
    y[1] = 2 * x[0] + 4 * x[1];
    return y;
  };

  LinearOperator A(2, 2, matvec, rmatvec);

  Vector x = Vector::Zero(2);
  x[0] = 1.0;
  x[1] = 0.0;

  std::cout << std::endl << "Abstract Linear Operators:" << std::endl;
  Vector y = A.matvec(x);
  std::cout << y[0] << ", " << y[1] << std::endl;

  y = A.rmatvec(x);
  std::cout << y[0] << ", " << y[1] << std::endl;

  LinearOperator B = A + A;
  Vector z = B.matvec(x);
  std::cout << z[0] << ", " << z[1] << std::endl;

  LinearOperator C = A * A;
  Vector zed = C.matvec(x);
  std::cout << zed[0] << ", " << zed[1] << std::endl;

  LinearOperator D = A.transpose();
  zed = D.matvec(x);
  std::cout << zed[0] << ", " << zed[1] << std::endl;

  std::vector<LinearOperator> vecs {A, A};
  LinearOperator E = block_diag(vecs);

  std::cout << "E " << E.m << ", " << E.n << std::endl;

  Vector in = Vector::Zero(4);
  Vector zed4 = E.rmatvec(in);
  std::cout << zed4[0] << ", " << zed4[1] << ", " << zed4[2] << ", " << zed4[3] << std::endl;

  std::cout << std::endl << "LSQR:" << std::endl;
  LsqrResult result = lsqr(A, x);
  std::cout << result.x[0] << ", " << result.x[1] << std::endl;
  print_result(result);

  Eigen::Matrix3d mat;
  mat << 1, 2, 3, 4, 5, 6, 7, 8, 9;
  std::cout << Eigen::Matrix3d(mat.triangularView<Eigen::Lower>()) << "\n\n";
}
