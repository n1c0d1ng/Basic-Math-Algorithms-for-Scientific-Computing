#include <iostream>
#include "matrizen.hpp"

int main() {
    Matrix A(3, 3);
    Matrix b(3, 1);
    
    A.data[0][0] = 10.0;
    A.data[0][1] = 1.0;
    A.data[0][2] = 2.0;
    A.data[1][0] = 1.0;
    A.data[1][1] = 10.0;
    A.data[1][2] = 1.0;
    A.data[2][0] = 2.0;
    A.data[2][1] = 1.0;
    A.data[2][2] = 10.0;

    b.data[0][0] = 1.0;
    b.data[1][0] = 2.0;
    b.data[2][0] = 1.0;

    //std::cout << "A is square: " << A.isSquare() << "\n";
    //std::cout << "b is square: " << b.isSquare() << "\n";
    Matrix x = A.JacobiIteration(b);
    //auto [L, U] = A.LUDecomposition();
    Matrix x_lu = A.LinSolve(b);
    //L.print("L matrix");
    //U.print("U matrix");
    x.print("Solution via Jacobi Iteration");
    std::cout << "----------------------------------------\n";
    x_lu.print("Solution via LU Decomposition");

    return 0;
}

