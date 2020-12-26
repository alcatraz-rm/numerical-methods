#include <fstream>
#include <iostream>
#include <cmath>

#include "matrix.h"
#include "householder_method.h"

using namespace std;

Matrix generate_matrix_exp(size_t height) {
    double alpha = 10;

    Matrix matrix(height);

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < height; j++) {
            matrix[i][j] = exp(-alpha * abs((i - j) * (i - j)));

        }
    }

    return matrix;
}

double* generate_matrix_exp_b(Matrix m) {

    double* b = new double[m.dim()];
    for (int i = 0; i < m.dim(); i++) {
        b[i] = 0;
        for (int j = 0; j < m.dim(); j++) {
            b[i] += m[i](j);
        }
    }

    return b;
}

void round(Matrix& M, double accuracy)
{
    for (size_t i = 0; i < M.dim(); i++)
    {
        for (size_t j = 0; j < M.dim(); j++)
        {
            if (fabs(M[i][j]) < accuracy)
            {
                M[i][j] = 0.0;
            }
        }
    }
}

void print_vector(double* b, size_t height) {
    for (int i = 0; i < height; i++) {
        cout << b[i] << " " << endl;
    }

}

double* multiply_b(Matrix M, double* b)
{
    double* result = new double[M.dim()];

    for (size_t i = 0; i < M.dim(); i++)
    {
        result[i] = 0.0;

        for (size_t j = 0; j < M.dim(); j++)
        {
            result[i] += M[i](j) * b[j];
        }
    }
    delete[] b;

    return result;
}


int main() {
    //ifstream fout("input.txt");
    size_t height = 10;

    //fout >> height;
    Matrix a(height);
    a = generate_matrix_exp(height);
    //fout >> a;

    double* b = new double[height];
    b = generate_matrix_exp_b(a);

    //for (int i = 0; i < height; i++) {
    //    fout >> b[i];
    //}

    //fout.close();

    for (int i = 0; i < height - 1; i++) {
        Matrix U(height - i);
        U = reflections(a, i);
        a = U * a;
        b = multiply_b(U, b);
    }

    //round(a, pow(10, -9));

    double* solution = express_variables(a, b);
    print_vector(solution, a.dim());

    delete[] b;
    delete[] solution;

    return 0;

}
