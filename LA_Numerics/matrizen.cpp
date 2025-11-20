// Matrizenklasse 
#include "matrizen.hpp"
#include <iostream>
#include <cmath>

bool Matrix::isSquare() const {
    return rows == cols;
}
Matrix Matrix::Multiply(const Matrix& other) const {
    if (cols != other.rows) {
        std::cout<< "Das Matrizenprodukt existiert nicht.\n";
        return Matrix(0, 0); // Funktion wird hier abgebrochen
    }
    Matrix result(rows, other.cols);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < other.cols; ++j) {
            for (int k = 0; k < cols; ++k) {
                result.data[i][j] += data[i][k] * other.data[k][j];
            }
        }
    }
    return result;
}

double Matrix::normVector() const {
    if (cols != 1) {
        std::cout << "Normenberechnung nur für Vektoren (n x 1 Matrizen) möglich.\n";
        return -1.0; // Fehlerwert
    }
    double sum = 0.0;
    for (int i = 0; i < rows; ++i) {
        sum += data[i][0]*data[i][0];
        }
    return std::sqrt(sum);
}

Matrix Matrix::JacobiIteration(const Matrix& other) const {
    // Implementierung der Jacobi-Iteration
    Matrix result(rows, other.cols);
    int iterations = 0;
    int maxIterations = std::pow(10, 6);
    double epsilon = 1e-5;
    Matrix x_old(rows, 1); // Initalisiert als Nullvektor
    double norm = 1.0;

    if (isSquare() == false) {
        std::cout << "Jacobi-Iteration erfordert ein quadratisches Problem.\n";
        return Matrix(rows, 0); // Fehlerhafte Matrix zurückgeben
    }

    while ((norm > epsilon) && (iterations < maxIterations)) {
    
        for (int i=0; i < rows; i++) {
            double sum = 0.0;
            for (int j=0; j < cols; j++) {
                if (j != i) {
                    sum += data[i][j] * x_old.data[j][0];
                }
            }
            result.data[i][0] = (other.data[i][0] - sum) / data[i][i];
        }
        norm = 0.0;
        for (int i = 0; i < rows; ++i) {
            norm += std::pow(result.data[i][0] - x_old.data[i][0], 2);
        }
        norm = std::sqrt(norm); 
        x_old = result;
        iterations ++;
    }
    return result;
};
