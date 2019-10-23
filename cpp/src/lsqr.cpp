#include "lsqr.h"
#include <assert.h>
#include <cmath>

inline double sign(const double &x) {
  if (x < 0) {
    return -1.0;
  } else if (x > 0) {
    return 1.0;
  } else {
    return 0.0;
  }
}

inline void _sym_ortho(double a, double b, double &c, double &s, double &r) {
  if (b == 0) {
    c = sign(a);
    s = 0.0;
    r = std::abs(a);
  } else if (a == 0) {
    c = 0.0;
    s = sign(b);
    r = std::abs(b);
  } else if (std::abs(b) > std::abs(a)) {
    double tau = a / b;
    s = sign(b) / std::sqrt(1 + tau * tau);
    c = s * tau;
    r = b / s;
  } else {
    double tau = b / a;
    c = sign(a) / std::sqrt(1 + tau * tau);
    s = c * tau;
    r = a / c;
  }
}

LsqrResult lsqr_sparse(const SparseMatrix &A, const Vector &b,
                       const double damp, const double atol, const double btol,
                       const double conlim, int iter_lim) {
  return lsqr(aslinearoperator(A), b, damp, atol, btol, conlim, iter_lim);
}

LsqrResult lsqr(const LinearOperator &A, const Vector &b, const double damp,
                const double atol, const double btol, const double conlim,
                int iter_lim) {
  int m = A.m;
  int n = A.n;

  if (iter_lim == -1) {
    iter_lim = 2 * A.n;
  }

  assert(iter_lim > 0);
  assert(b.size() == m);
  assert(damp >= 0.0);
  assert(atol >= 0.0);
  assert(btol >= 0.0);
  assert(conlim >= 0.0);

  Vector var = Vector::Zero(n);

  int itn = 0;
  int istop = 0;
  double ctol = 0.0;
  if (conlim > 0) {
    ctol = 1.0 / conlim;
  }
  double anorm = 0.0;
  double acond = 0.0;
  double dampsq = damp * damp;
  double ddnorm = 0.0;
  double res2 = 0.0;
  double xnorm = 0.0;
  double xxnorm = 0.0;
  double z = 0.0;
  double cs2 = -1.0;
  double sn2 = 0.0;

  Vector _xm = Vector::Zero(m);
  Vector _xn = Vector::Zero(n);
  Vector v = Vector::Zero(n);
  Vector u = b;
  Vector x = Vector::Zero(n);
  double alfa = 0.0;
  double beta = u.norm();
  Vector w = Vector::Zero(n);
  Vector dk = Vector::Zero(n);

  if (beta > 0) {
    u /= beta;
    v = A.rmatvec(u);
    alfa = v.norm();
  }

  if (alfa > 0) {
    v /= alfa;
    w = v;
  }

  double rhobar = alfa;
  double phibar = beta;
  double bnorm = beta;
  double rnorm = beta;
  double r1norm = rnorm;
  double r2norm = rnorm;

  double arnorm = alfa * beta;

  double rhobar1 = 0.0;
  double cs1 = 0.0;
  double sn1 = 0.0;
  double psi = 0.0;
  double cs = 0.0;
  double sn = 0.0;
  double rho = 0.0;
  double theta = 0.0;
  double phi = 0.0;
  double tau = 0.0;
  double t1 = 0.0;
  double t2 = 0.0;
  double delta = 0.0;
  double gambar = 0.0;
  double rhs = 0.0;
  double zbar = 0.0;
  double gamma = 0.0;
  double res1 = 0.0;
  double r1sq = 0.0;
  double test1 = 0.0;
  double test2 = 0.0;
  double test3 = 0.0;
  double rtol = 0.0;

  // The exact solution is x = 0
  if (arnorm == 0) {
    iter_lim = -1;
  }

  while (itn < iter_lim) {
    itn += 1;
    u = A.matvec(v) - alfa * u;
    beta = u.norm();

    if (beta > 0) {
      u /= beta;
      anorm = std::sqrt(anorm * anorm + alfa * alfa + beta * beta + damp * damp);
      v = A.rmatvec(u) - beta * v;
      alfa = v.norm();
      if (alfa > 0) {
        v /= alfa;
      }
    }

    rhobar1 = std::sqrt(rhobar * rhobar + damp * damp);
    cs1 = rhobar / rhobar1;
    sn1 = damp / rhobar1;
    psi = sn1 * phibar;
    phibar = cs1 * phibar;

    // givens rotation
    _sym_ortho(rhobar1, beta, cs, sn, rho);

    theta = sn * alfa;
    rhobar = -cs * alfa;
    phi = cs * phibar;
    phibar = sn * phibar;
    tau = sn * phi;

    t1 = phi / rho;
    t2 = -theta / rho;
    dk = w / rho;

    x = x + t1 * w;
    w = v + t2 * w;
    ddnorm = ddnorm + dk.squaredNorm();

    delta = sn2 * rho;
    gambar = -cs2 * rho;
    rhs = phi - delta * z;
    zbar = rhs / gambar;
    xnorm = std::sqrt(xxnorm + zbar * zbar);
    gamma = std::sqrt(gambar * gambar + theta * theta);
    cs2 = gambar / gamma;
    sn2 = theta / gamma;
    z = rhs / gamma;
    xxnorm = xxnorm + z * z;

    acond = anorm * std::sqrt(ddnorm);
    res1 = phibar * phibar;
    res2 = res2 + psi * psi;
    rnorm = std::sqrt(res1 + res2);
    arnorm = alfa * std::abs(tau);

    r1sq = rnorm * rnorm - dampsq * xxnorm;
    r1norm = std::sqrt(std::abs(r1sq));
    if (r1sq < 0) {
      r1norm = -r1norm;
    }
    r2norm = rnorm;

    test1 = rnorm / bnorm;
    test2 = arnorm / (anorm * rnorm);
    test3 = 1.0 / acond;
    t1 = test1 / (1 + anorm * xnorm / bnorm);
    rtol = btol + atol * anorm * xnorm / bnorm;

    if (itn >= iter_lim) {
      istop = 7;
    }
    if (1 + test3 <= 1) {
      istop = 6;
    }
    if (1 + test2 <= 1) {
      istop = 5;
    }
    if (1 + t1 <= 1) {
      istop = 4;
    }
    if (test3 <= ctol) {
      istop = 3;
    }
    if (test2 <= atol) {
      istop = 2;
    }
    if (test1 <= rtol) {
      istop = 1;
    }

    if (istop != 0) {
      break;
    }
  }

  LsqrResult result = {x,     istop, itn,    r1norm, r2norm,
                       anorm, acond, arnorm, xnorm};

  return result;
}
