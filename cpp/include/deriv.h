#pragma once

#include "cones.h"
#include "eigen_includes.h"
#include "linop.h"

LinearOperator M_operator(const SparseMatrix &Q, const std::vector<Cone> &cones,
                          const Vector &u, const Vector &v, double w);
