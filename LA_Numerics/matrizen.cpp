// Matrizenklasse 
#include "matrizen.hpp"
#include <iostream>
#include <cmath>

bool Matrix::isSquare() const {
    return rows == cols;
}

void Matrix::print(const std::string& name) {
    std::cout << name << ":" << std::endl;
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            std::cout << data[i][j] << " ";
        }
        std::cout << std::endl;
    }
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

std::pair<Matrix, Matrix> Matrix::LUDecomposition() const {
    // Implementierung der LU-Zerlegung
    Matrix L(rows, cols);
    Matrix U(rows, cols);

    for (int i = 0; i < rows; ++i) {
        // U_ij = A_ij - sum_{k=0}^{i-1} L_ik * U_kj
        for (int j = i; j < cols; ++j) {
            U.data[i][j] = data[i][j];
            for (int k = 0; k < i; ++k) {
                U.data[i][j] -= L.data[i][k] * U.data[k][j];
            }
        }
        // L_ji = (A_ji - sum_{k=0}^{i-1} L_jk * U_ki) / U_ii
        // L_ii = 1
        for (int j = i; j < rows; ++j) {
            if (i == j)
                L.data[i][i] = 1; 
            else {
                L.data[j][i] = data[j][i];
                for (int k = 0; k < i; ++k) {
                    L.data[j][i] -= L.data[j][k] * U.data[k][i];
                }
                L.data[j][i] /= U.data[i][i];
            }
        }
    }
    return std::make_pair(L, U);
};

Matrix Matrix::LinSolve(const Matrix& b) const {
    // Implementierung der Lösung eines linearen Gleichungssystems Ax = b
    // mittels LU-Zerlegung und Vorwärts-/Rückwärtseinsetzen
    auto [L, U] = LUDecomposition();
    Matrix y(rows, 1);
    Matrix x(rows, 1);

    // Vorwärtseinsetzen Ly = b
    for (int i = 0; i < rows; ++i) {
        y.data[i][0] = b.data[i][0];
        for (int j = 0; j < i; ++j) {
            y.data[i][0] -= L.data[i][j] * y.data[j][0];
        }
    }

    // Rückwärtseinsetzen Ux = y
    for (int i = rows - 1; i >= 0; --i) {
        x.data[i][0] = y.data[i][0];
        for (int j = i + 1; j < cols; ++j) {
            x.data[i][0] -= U.data[i][j] * x.data[j][0];
        }
        x.data[i][0] /= U.data[i][i];
    }
    return x;
};


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
