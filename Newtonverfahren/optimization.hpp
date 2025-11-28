#include "matrizen.hpp"
#include <functional>

// Sammlung mathematischer Algorithmen für Optimierung

Matrix CG_Verfahren(const Matrix &A, const Matrix &b);

Matrix GradientDescent(
    const Matrix &start, 
    std::function<double(const Matrix&)> func, 
    std::function<Matrix(const Matrix&)> grad
);
Matrix NewtonMethod(
    const Matrix &start, 
    std::function<double(const Matrix&)> func, 
    std::function<Matrix(const Matrix&)> grad, 
    std::function<Matrix(const Matrix&)> hess
);