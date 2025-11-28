#include "Rosenbrock.hpp"

const double a = 1.0;
const double b = 100.0;

double RosenbrockFunction(Matrix const& x) {
    // Rosenbrock-Funktion: f(x,y) = (a - x)^2 + b*(y - x^2)^2
    // Standardwerte: a = 1, b = 100
    double term1 = a - x.data[0][0];
    double term2 = x.data[1][0] - x.data[0][0] * x.data[0][0];
    return term1 * term1 + b * term2 * term2;
}

Matrix RosenbrockGradient(Matrix const& x) {
    // Gradient der Rosenbrock-Funktion
    // df/dx = -2(a - x) - 4b*x(y - x^2)
    // df/dy = 2b(y - x^2)
    double dfdx = -2.0 * (a - x.data[0][0]) - 4.0 * b * x.data[0][0] * (x.data[1][0] - x.data[0][0] * x.data[0][0]);
    double dfdy = 2.0 * b * (x.data[1][0] - x.data[0][0] * x.data[0][0]);
    Matrix grad(2, 1);
    grad.data[0][0] = dfdx;
    grad.data[1][0] = dfdy;
    return grad;
}

Matrix RosenbrockHessian(Matrix const& x) {
    // Hesse-Matrix der Rosenbrock-Funktion
    // d2f/dx2 = 2 - 4b(y - x^2) + 8b*x^2
    // d2f/dy2 = 2b
    // d2f/dxdy = -4b*x
    double d2fdx2 = 2.0 - 4.0 * b * (x.data[1][0] - x.data[0][0] * x.data[0][0]) + 8.0 * b * x.data[0][0] * x.data[0][0];
    double d2fdy2 = 2.0 * b;
    double d2fdxdy = -4.0 * b * x.data[0][0];

    Matrix hess(2, 2);
    hess.data[0][0] = d2fdx2;
    hess.data[0][1] = d2fdxdy;
    hess.data[1][0] = d2fdxdy;
    hess.data[1][1] = d2fdy2;
    return hess;
}