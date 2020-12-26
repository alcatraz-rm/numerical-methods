#include <fstream>
#include <iostream>

#include "matrix.h"
#include "inverse_matrix.h"
#include "gauss_seidel_method.h"

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

int main()
{
    //ifstream fout("input.txt");
    size_t height = 5;

    //fout >> height;
    Matrix a(height);
    a = generate_matrix_exp(height);
    //fout >> a;

    double* b = new double[height];
    b = generate_matrix_exp_b(height);

    /*for (int i = 0; i < height; i++)
    {
        fout >> b[i];
    }

    fout.close();*/

    double* solution = solve(a, b);
    cout << endl;

    for (int i = 0; i < height; i++)
    {
        cout << solution[i] << endl;
    }
    //cout << inverse(a);

    delete[] b;
    delete[] solution;
    return 0;
}
