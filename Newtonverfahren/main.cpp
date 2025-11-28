// Main Programm zum CG Verfahren in C++
#include "matrizen.hpp"
#include "Rosenbrock.hpp"
#include "optimization.hpp"
#include <iostream>

int main() {
    // AUfruf CG verfahren
    Matrix start_vector(2,1);
    start_vector.data[0][0] = -10.2;
    start_vector.data[1][0] = 5.0;

    Matrix grad = RosenbrockGradient(start_vector);
    Matrix hess = RosenbrockHessian(start_vector);

    //Lösung des Gleichungssystems H * x = -grad
    Matrix x = CG_Verfahren(hess, (-1)*grad);
    Matrix Kontrollrechnung = hess.Multiply(x) + grad;
    //x.print("Lösung des Gleichungssystems");
    //Kontrollrechnung.print("Kontrollrechnung (sollte nahe 0 sein)");
    Matrix solution_GD = GradientDescent(
        start_vector,
        RosenbrockFunction,
        RosenbrockGradient
    );


    Matrix solution_NM = NewtonMethod(
        start_vector,
        RosenbrockFunction,
        RosenbrockGradient,
        RosenbrockHessian
    );

    solution_NM.print("Lösung via Newton-Verfahren");
    std::cout << "-----------------------\n";
    solution_GD.print("Lösung via Gradientenverfahren");
    
    return 0;
}