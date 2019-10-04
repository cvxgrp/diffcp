#pragma once

// #include "linop.h"

// struct LsqrResult {
//   Vector x; // the solution
//   int istop; // reason for termination
//   int itn; // num iterations
//   double r1norm; // norm(b-Ax)
//   double r2norm; // sqrt(norm(b-Ax)^2 + damp^2 * norm(x)^2)
//   double anorm; // Estimate of Frobenius norm of Abar = (A, damp * I)
//   double acond; // Estimate of cond(Abar)
//   double arnorm; // Estimate of norm(A'*(b-Ax) - damp^2*x)
//   double xnorm; // norm(x)
// };

// /**
//  * Find the least-squares solution to an abstract linear system of equations.
//  * This function solves the optimization problem
//  *  minimize ||Ax-b||^2 + damp^2||x||^2
//  */
// LsqrResult lsqr(LinearOperator A, Vector b, double damp=0.0, double atol=1e-8, double btol=1e-8, double conlim=1e8,
//             int iter_lim=-1, bool show=false);