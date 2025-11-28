// Konkrete Implementierung der Optimierungsalgorithmen
#include "optimization.hpp"
#include <iostream>

Matrix CG_Verfahren(const Matrix &A, const Matrix &b)
{
    Matrix r = b; // Anfangsresiduum r = b - Ax, hier A*x=0
    Matrix d = r; // Anfangsrichtung d = r

    int maxIterations = 1e3;
    double epsilon = 1e-10;

    // 1 Startpunkt: x0 = 0
    Matrix x = Matrix(A.rows, 1);
    Matrix residuum = b;  // r_0 = b
    Matrix dk = residuum; // d_0 = r_0

    for (int k = 0; k < maxIterations; k++)
    {
        // 2 Update der Lösung qk = A d_k
        Matrix qk = A.Multiply(dk);
        // alpha_k = dk*rk / dk*qk
        double alpha = dot(dk, residuum) / dot(dk, qk);
        // Iterationsschritt: x_k+1 = x_k + alpha_k * d_k
        x = x + alpha * dk;
        // 3 Update des Residuums rk+1 = rk - alpha_k * qk
        residuum = residuum - (alpha * qk);
        if (residuum.normVector() < epsilon)
        {
            return x;
        }
        // beta_k = - (rk+1 * qk) / (dk * qk)
        double beta = -(dot(residuum, qk)) / (dot(dk, qk));
        // 4 Berechnung des neuen Suchvektors dk+1 = rk+1 + beta_k * dk
        dk = residuum + beta * dk;
    }
    std::cout << "Warnung: Maximale Iterationszahl erreicht, keine Konvergenz!\n";
    return x;
};

// Newton Schritt d_k = - H(x_k)^-1 * grad(x_k) <-> H(x_k) * d_k = - grad(x_k)
Matrix NewtonMethod(
    const Matrix &start, 
    std::function<double(const Matrix&)> func, 
    std::function<Matrix(const Matrix&)> grad, 
    std::function<Matrix(const Matrix&)> hess
) {
    Matrix x = start;
    int maxIterations = 100;
    double epsilon = 1e-6;

    for (int i = 0; i < maxIterations; ++i) {
        Matrix gradient = grad(x);
        if (gradient.normVector() < epsilon) {
            std::cout << "Newton-Verfahren konvergiert nach " << i << " Iterationen.\n";
            return x;
        }
        Matrix hessian = hess(x);
        Matrix p = CG_Verfahren(hessian, (-1) * gradient);
        x = x + p;
    }
    std::cout << "Warnung: Maximale Iterationszahl erreicht, keine Konvergenz!\n";
    return x;
}

Matrix GradientDescent(
    const Matrix &start, 
    std::function<double(const Matrix&)> func, 
    std::function<Matrix(const Matrix&)> grad
) {
    Matrix x = start;
    int maxIterations = 1e6;
    double epsilon = 1e-6;

    for (int i = 0; i < maxIterations; ++i) {
        Matrix gradient = grad(x);
        if (gradient.normVector() < epsilon) {
            std::cout << "Gradientenverfahren konvergiert nach " << i << " Iterationen.\n";
            return x;
        }
        // Armijo Schrittweitenregel: nabla f(x - alpha * grad) < nabla f(x) - c * alpha * ||grad||^2
        double alpha = 1.0;
        double c = 1e-4;
        while (func(x - alpha * gradient) > func(x) - c * alpha * dot(gradient, gradient)) {
            alpha *= 0.5; // Schrittweite verkleinern
            if (alpha < 1e-10) {
                break; // Vermeidung zu kleiner Schrittweiten
            }
        }
        x = x - alpha * gradient;
    }
    std::cout << "Warnung: Maximale Iterationszahl erreicht, keine Konvergenz!\n";
    return x;
}