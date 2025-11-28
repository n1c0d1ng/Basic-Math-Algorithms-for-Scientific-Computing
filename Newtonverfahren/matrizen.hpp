#pragma once
#include <vector>
#include <string>

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
    Matrix LinSolve(const Matrix& b) const;
    std::pair<Matrix, Matrix> LUDecomposition() const;
    double normVector() const;
    void print(const std::string& name);
};

// Freie Funktion außerhalb der Klasse:
Matrix operator*(const Matrix &A, const Matrix &B);
Matrix operator*(double scalar, const Matrix& vector);
Matrix operator+(const Matrix& A, const Matrix& B);
Matrix operator-(const Matrix &A, const Matrix &B);
double dot(const Matrix& u, const Matrix& v);