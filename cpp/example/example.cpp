#include "linop.h"
#include <iostream>

int main() {
  const std::function<Vector(const Vector &)> matvec =
      [](const Vector &x) -> Vector {
    Vector y = Vector::Zero(2);
    y[0] = 1 * x[0] + 2 * x[1];
    y[1] = 3 * x[0] + 4 * x[1];
    return y;
  };

  const auto rmatvec = [](const Vector &x) {
    Vector y = Vector::Zero(2);
    y[0] = 1 * x[0] + 3 * x[1];
    y[1] = 2 * x[0] + 4 * x[1];
    return y;
  };

  LinearOperator A(2, 2, matvec, rmatvec);

  Vector x = Vector::Zero(2);
  x[0] = 1.0;
  x[1] = 0.0;

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
}
