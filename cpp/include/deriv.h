#pragma once

#include "linop.h"
#include "cones.h"
#include "lsqr.h"

Vector _solve_derivative(ListOfCones cones, Vector z, Vector rhs);
Vector _solve_adjoint_derivative(ListOfCones cones, Vector z, Vector rhs);