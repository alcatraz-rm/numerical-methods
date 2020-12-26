#include <iostream>

#include "matrix.h"

using namespace std;


double scalar_product(double* a, double* b, size_t height) {
    double sum = 0;
    for (int i = 0; i < height; i++) {
        sum += a[i] * b[i];
    }
    return sum;
}

double getLength(double* b, size_t height) {
    return sqrt(scalar_product(b, b, height));
}

double* express_variables(Matrix A, double* b)
{
    double* solution = new double[A.dim()];
    double value;

    for (int i = A.dim() - 1; i >= 0; i--)
    {
        if (A[i][i] != 0)
        {
            value = b[i];

            for (size_t j = i + 1; j < A.dim(); j++)
            {
                value -= A[i][j] * solution[j];
            }

            solution[i] = value / A[i][i];
        }
        else
        {
            cout << "Can't solve (more than 1 solution or 0)." << endl;
            delete[] solution;

            return nullptr;
        }

    }

    return solution;
}

Matrix reflections(Matrix a, size_t i1) {
    Matrix tmp1(a.dim());
    tmp1 = a;

    for (int i = 0; i < i1; i++) {
        tmp1 = tmp1(0, 0);
    }

    double* b = new double[tmp1.dim()];

    for (int i = 0; i < tmp1.dim(); i++) {
        b[i] = tmp1[i][0];
    }

    double b_length = getLength(b, tmp1.dim());
    double* b_ac = new double[tmp1.dim()];

    for (int i = 1; i < tmp1.dim(); i++) {
        b_ac[i] = b[i];
    }

    b_ac[0] = b[0] - b_length;

    double b_b_ac = scalar_product(b_ac, b, tmp1.dim());

    double* w = new double[tmp1.dim()];

    for (int i = 0; i < tmp1.dim(); i++) {
        w[i] = b_ac[i] / sqrt(2 * b_b_ac);
    }

    Matrix uut(tmp1.dim());

    for (int i = 0; i < tmp1.dim(); i++) {
        for (int j = 0; j < tmp1.dim(); j++) {
            uut[i][j] = w[i] * w[j];
        }
    }

    Matrix E(tmp1.dim());
    E = E - 2 * uut;
    Matrix result(a.dim());

    size_t ei = 0;
    size_t ej = 0;

    for (int i = a.dim() - tmp1.dim(); i < a.dim(); i++) {
        for (int j = a.dim() - tmp1.dim(); j < a.dim(); j++) {
            result[i][j] = E[i - i1][j - i1];
        }
    }

    return result;
}
