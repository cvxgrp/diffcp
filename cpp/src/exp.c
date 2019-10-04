#include "exp.h"
#include <math.h>
#include <string.h>

#define CONE_TOL (1e-8)
#define CONE_THRESH (1e-6)
#define EXP_CONE_MAX_ITERS (200)

#define ABS(x) (((x) < 0) ? -(x) : (x))
#define MAX(a, b) (((a) > (b)) ? (a) : (b))

double exp_newton_one_d(double rho, double y_hat,
                                  double z_hat) {
  double t = MAX(-z_hat, 1e-6);
  double f, fp;
  int i;
  for (i = 0; i < EXP_CONE_MAX_ITERS; ++i) {
    f = t * (t + z_hat) / rho / rho - y_hat / rho + log(t / rho) + 1;
    fp = (2 * t + z_hat) / rho / rho + 1 / t;

    t = t - f / fp;

    if (t <= -z_hat) {
      return 0;
    } else if (t <= 0) {
      return z_hat;
    } else if (ABS(f) < CONE_TOL) {
      break;
    }
  }
  return t + z_hat;
}

void exp_solve_for_x_with_rho(double *v, double *x,
                                     double rho) {
  x[2] = exp_newton_one_d(rho, v[1], v[2]);
  x[1] = (x[2] - v[2]) * x[2] / rho;
  x[0] = v[0] - rho;
}

double exp_calc_grad(double *v, double *x, double rho) {
  exp_solve_for_x_with_rho(v, x, rho);
  if (x[1] <= 1e-12) {
    return x[0];
  }
  return x[0] + x[1] * log(x[1] / x[2]);
}

void exp_get_rho_ub(double *v, double *x, double *ub,
                           double *lb) {
  *lb = 0;
  *ub = 0.125;
  while (exp_calc_grad(v, x, *ub) > 0) {
    *lb = *ub;
    (*ub) *= 2;
  }
}

/* project onto the exponential cone, v has dimension *exactly* 3 */
int _proj_exp_cone(double *v, double *rho) {
  int i;
  double ub, lb, g, x[3];
  double r = v[0], s = v[1], t = v[2];
  double tol = CONE_TOL; /* iter < 0 ? CONE_TOL : MAX(CONE_TOL, 1 /
                               POWF((iter + 1), CONE_RATE)); */

  /* v in cl(Kexp) */
  if ((s * exp(r / s) - t <= CONE_THRESH && s > 0) ||
      (r <= 0 && s == 0 && t >= 0)) {
    return 0;
  }

  /* -v in Kexp^* */
  if ((-r < 0 && r * exp(s / r) + exp(1) * t <= CONE_THRESH) ||
      (-r == 0 && -s >= 0 && -t >= 0)) {
    memset(v, 0, 3 * sizeof(double));
    return 0;
  }

  /* special case with analytical solution */
  if (r < 0 && s < 0) {
    v[1] = 0.0;
    v[2] = MAX(v[2], 0);
    return 0;
  }

  /* iterative procedure to find projection, bisects on dual variable: */
  exp_get_rho_ub(v, x, &ub, &lb); /* get starting upper and lower bounds */
  for (i = 0; i < EXP_CONE_MAX_ITERS; ++i) {
    *rho = (ub + lb) / 2;          /* halfway between upper and lower bounds */
    g = exp_calc_grad(v, x, *rho); /* calculates gradient wrt dual var */
    if (g > 0) {
      lb = *rho;
    } else {
      ub = *rho;
    }
    if (ub - lb < tol) {
      break;
    }
  }

  v[0] = x[0];
  v[1] = x[1];
  v[2] = x[2];
  return 0;
}