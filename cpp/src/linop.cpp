#include "linop.h"

LinearOperator::LinearOperator(
    int rows,
    int cols,
    const std::function<Vector(const Vector&)>& matvec_in,
    const std::function<Vector(const Vector&)>& rmatvec_in) {
    m = rows;
    n = cols;
    matvec = matvec_in;
    rmatvec = rmatvec_in;
}


LinearOperator LinearOperator::operator + (LinearOperator const& obj)) {
    const auto result_matvec = [](const Vector& x) {
        return matvec(x) + obj.matvec(x);
    }
    const auto result_rmatvec = [](const Vector& x) {
        return rmatvec(x) + obj.rmatvec(x);
    }
    LinearOperator result(m, n, result_matvec, result_rmatvec);
    return result;
}


LinearOperator LinearOperator::operator - (LinearOperator const& obj) {
    const auto result_matvec = [](const Vector& x) {
        return matvec(x) + obj.matvec(-x);
    }
    const auto result_rmatvec = [](const Vector& x) {
        return rmatvec(x) + obj.rmatvec(-x);
    }
    LinearOperator result(m, n, result_matvec, result_rmatvec);
    return result;
}


LinearOperator LinearOperator::operator * (LinearOperator const& obj) {
    const auto result_matvec = [](const Vector& x) {
        return matvec(obj.matvec(x));
    }
    const auto result_rmatvec = [](const Vector& x) {
        return obj.rmatvec(rmatvec(x));
    }
    LinearOperator result(m, n, result_matvec, result_rmatvec);
    return result;
}


LinearOperator block_diag(const std::vector<LinearOperator>& linear_operators) {
    int rows = 0;
    int cols = 0;

    for (auto linop : linear_operators) {
        rows += linop.m;
        cols += linop.n;
    }

    const auto result_matvec = [](const Vector& x) {
        auto out = Vector::Zero(rows);
        int i = 0;
        int j = 0;
        for (auto linop : linear_operators) {
            out.segment(i, i + linop.m) = linop.matvec(x.segment(j, j + linop.n));
            i += linop.m;
            j += linop.n;
        }
        return out;
    }
    const auto result_rmatvec = [](const Vector& x) {
        auto out = Vector::Zero(cols);
        int i = 0;
        int j = 0;
        for (auto linop : linear_operators) {
            out.segment(i, i + linop.n) = linop.rmatvec(x.segment(j, j + linop.m));
            i += linop.n;
            j += linop.m;
        }
        return out;
    }

    LinearOperator result(rows, cols, result_matvec, result_rmatvec);

    return result;
}