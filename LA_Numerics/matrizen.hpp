#pragma once
#include <vector>

class Matrix {
    public:
    int rows;
    int cols;
    std::vector<std::vector<double>> data;
    //public:
    Matrix(int m, int n):
        // Initialisierungsliste
        rows(m),
        cols(n), 
        data(m, std::vector<double>(n)) 
        {
        // Initialisierung der Matrix mit Nullen
    }
    bool isSquare() const;
    Matrix Multiply(const Matrix& other) const;
    Matrix JacobiIteration(const Matrix& other) const;
    double normVector() const;
};