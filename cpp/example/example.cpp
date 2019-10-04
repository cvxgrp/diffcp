#include "linop.h"
#include <iostream>

int main() {
    const auto matvec = [](const Vector& x) {
        Vector y = Vector::Zero(2);
        y[0] = 1 * x[0] + 2 * x[1];
        y[1] = 3 * x[0] + 4 * x[1];
        return y;
    };
    
    const auto rmatvec = [](const Vector& x) {
        Vector y = Vector::Zero(2);
        y[0] = 1 * x[0] + 3 * x[1];
        y[1] = 2 * x[0] + 4 * x[1];
        return y;
    };

    LinearOperator A;
    A.m = 2;
    A.n = 2;
    A.matvec = matvec;
    A.rmatvec = rmatvec;

    Vector x = Vector::Zero(2);
    x[0] = 1.0;
    x[1] = 0.0;

    auto y = A.matvec(x);
    std::cout << y[0] << ", " << y[1] << std::endl;

    y = A.rmatvec(x);
    std::cout << y[0] << ", " << y[1] << std::endl;

    auto B = A + A;
}