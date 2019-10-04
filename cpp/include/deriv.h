#pragma once

#include "linop.h"
#include "cones.h"

Vector _solve_derivative(const SparseMatrix& Q, const ListOfCones cones, const Vector& z, const Vector& rhs);
Vector _solve_adjoint_derivative(const SparseMatrix& Q, const ListOfCones cones, const Vector& z, const Vector& dz);