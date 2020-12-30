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


Matrix iteration(Matrix A)
{
    Matrix Q(A.dim());
    Matrix R(A.dim());
    Matrix H(A.dim());
    R = A;

    for (size_t i = 0; i < A.dim() - 1; i++)
    {
        H = reflections(R, i);
        R = H * R;
        Q *= H;
    }

    //cout << Q;
    //cout << endl;
    //cout << R;
    //cout << endl;

    cout << R * Q;
    cout << endl << endl;

    return R * Q;
}


double* solve(Matrix A)
{
    for (size_t i = 0; i < 11; i++)
    {
        A = iteration(A);
    }

    cout << A;
    return 0;
}
