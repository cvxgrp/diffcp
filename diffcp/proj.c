// TODO(akshayka): types ...
// TODO(akshayka): license ...
// TODO(akshayka): Python glue ...
#include <Python.h>
#include <math.h>

typedef double scs_float;
typedef int scs_int;

#define CONE_TOL (1e-8)
#define CONE_THRESH (1e-6)
#define EXP_CONE_MAX_ITERS (100)

#define ABS(x) (((x) < 0) ? -(x) : (x))
#define MAX(a, b) (((a) > (b)) ? (a) : (b))


static scs_float exp_newton_one_d(scs_float rho, scs_float y_hat,
                                  scs_float z_hat) {
  scs_float t = MAX(-z_hat, 1e-6);
  scs_float f, fp;
  scs_int i;
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

static void exp_solve_for_x_with_rho(scs_float *v, scs_float *x,
                                     scs_float rho) {
  x[2] = exp_newton_one_d(rho, v[1], v[2]);
  x[1] = (x[2] - v[2]) * x[2] / rho;
  x[0] = v[0] - rho;
}

static scs_float exp_calc_grad(scs_float *v, scs_float *x, scs_float rho) {
  exp_solve_for_x_with_rho(v, x, rho);
  if (x[1] <= 1e-12) {
    return x[0];
  }
  return x[0] + x[1] * log(x[1] / x[2]);
}

static void exp_get_rho_ub(scs_float *v, scs_float *x, scs_float *ub,
                           scs_float *lb) {
  *lb = 0;
  *ub = 0.125;
  while (exp_calc_grad(v, x, *ub) > 0) {
    *lb = *ub;
    (*ub) *= 2;
  }
}

/* project onto the exponential cone, v has dimension *exactly* 3 */
static scs_int _proj_exp_cone(scs_float *v, scs_float *rho) {
  scs_int i;
  scs_float ub, lb, g, x[3];
  scs_float r = v[0], s = v[1], t = v[2];
  scs_float tol = CONE_TOL; /* iter < 0 ? CONE_TOL : MAX(CONE_TOL, 1 /
                               POWF((iter + 1), CONE_RATE)); */

  /* v in cl(Kexp) */
  if ((s * exp(r / s) - t <= CONE_THRESH && s > 0) ||
      (r <= 0 && s == 0 && t >= 0)) {
    return 0;
  }

  /* -v in Kexp^* */
  if ((-r < 0 && r * exp(s / r) + exp(1) * t <= CONE_THRESH) ||
      (-r == 0 && -s >= 0 && -t >= 0)) {
    memset(v, 0, 3 * sizeof(scs_float));
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
  /*
#if EXTRA_VERBOSE > 0
  scs_printf("exponential cone proj iters %i\n", i);
#endif
   */
  v[0] = x[0];
  v[1] = x[1];
  v[2] = x[2];
  return 0;
}

/* Returns a list with four floats, r, s, t (the variables in the cone), and
   rho, the dual variable. */
static PyObject* proj_exp_cone(PyObject *self, PyObject *args) {
  scs_float r;
  scs_float s;
  scs_float t;

  if (!PyArg_ParseTuple(args, "ddd", &r, &s, &t)) {
    return NULL;
  }
  scs_float v[3];
  scs_float rho;
  v[0] = r;
  v[1] = s;
  v[2] = t;
  _proj_exp_cone(v, &rho);
  return Py_BuildValue("dddd", v[0], v[1], v[2], rho);
}

/*  define functions in module */
static PyMethodDef ProjMethods[] = {
  {"proj_exp_cone", proj_exp_cone, METH_VARARGS,
    "Project onto the exponential cone."},
  {NULL, NULL, 0, NULL}
};

#if PY_MAJOR_VERSION >= 3
/* module initialization */
/* Python version 3*/
static struct PyModuleDef projmodule = {
  PyModuleDef_HEAD_INIT,
  "_proj", "C implementations of projections.",
  -1,
  ProjMethods
};

PyMODINIT_FUNC
PyInit__proj(void) {
  return PyModule_Create(&projmodule);
}

#else

/* module initialization */
/* Python version 2 */
PyMODINIT_FUNC
initproj_module(void) {
  (void) Py_InitModule("_proj", ProjMethods);
}
#endif
