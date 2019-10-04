#pragma once

#include <vector>

#include "eigen_includes.h"
#include "linop.h"

enum ConeType { ZERO, POS, SOC, PSD, EXP };

typedef struct Cone {
  ConeType type;
  std::vector<int> sizes;
} Cone;

/* Compute the derivative, at `x`, of a projection onto a cone.
 *
 *  Args:
 *    x:     The point at which to evaluate the derivative
 *    cones: A list of cones; the cone on which to project is the cartesian
 *           product of these cones
 *    dual:  whether to project onto the dual cone
 *
 *  Returns:
 *    A linear operator representing the derivative, at x, of the projection
 *    onto the cone.
 */
LinearOperator dprojection(Vector x, std::vector<Cone> cones, bool dual);
