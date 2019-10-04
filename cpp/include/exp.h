#pragma once

double exp_newton_one_d(double rho, double y_hat,
                                  double z_hat);

void exp_solve_for_x_with_rho(double *v, double *x,
                                     double rho);

double exp_calc_grad(double *v, double *x, double rho);

void exp_get_rho_ub(double *v, double *x, double *ub,
                           double *lb);

/* project onto the exponential cone, v has dimension *exactly* 3 */
int _proj_exp_cone(double *v, double *rho);