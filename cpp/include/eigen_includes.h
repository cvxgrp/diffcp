#pragma once

#include "Eigen/Dense"
#include "Eigen/Sparse"

using Vector = Eigen::VectorXd;
using Array = Eigen::Array<double, Eigen::Dynamic, 1>;
using Matrix = Eigen::MatrixXd;
using MatrixRef = Eigen::Ref<Eigen::MatrixXd>;
using SparseMatrix = Eigen::SparseMatrix<double>;
