#pragma once

#include "eigen_includes.h"
#include <functional>
#include <vector>

class LinearOperator {
    /**
     * m x n linear operator
    */
    public:
        int m;
        int n;
        std::function<Vector(const Vector&)> matvec;
        std::function<Vector(const Vector&)> rmatvec;

        LinearOperator() {};
        LinearOperator(
            int rows,
            int cols,
            const std::function<Vector(const Vector&)>& matvec_in,
            const std::function<Vector(const Vector&)>& rmatvec_in);
        LinearOperator operator+(LinearOperator const& obj);
        LinearOperator operator-(LinearOperator const& obj);
        LinearOperator operator*(LinearOperator const& obj);
        void transpose() {return std::swap(matvec, rmatvec);}
};

LinearOperator block_diag(const std::vector<LinearOperator>& linear_operators);

