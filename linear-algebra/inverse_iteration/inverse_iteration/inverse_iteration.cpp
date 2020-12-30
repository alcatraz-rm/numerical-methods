#include <iostream>
using namespace std;

#include "matrix.h"
#include "inverse_matrix.h"


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


pair<double*, double> iteration(Matrix A, double* x_prev, double mu)
{
    double u;

    for (size_t i = 0; i < A.dim(); i++)
    {
        A[i][i] -= mu;
    }

    Matrix B(A.dim());
    B = inverse(A);

    double* r = new double[A.dim()];
    for (size_t i = 0; i < A.dim(); i++)
    {
        r[i] = 0;
    }

    for (size_t i = 0; i < A.dim(); i++)
    {
        for (size_t j = 0; j < A.dim(); j++)
        {
            r[i] += A[i](j) * x_prev[j];
        }
    }
    double c = getLength(r, A.dim());

    for (size_t i = 0; i < A.dim(); i++)
    {
        r[i] /= c;
    }

    double* Ar = new double[A.dim()];
    for (size_t i = 0; i < A.dim(); i++)
    {
        Ar[i] = 0;
    }

    for (size_t i = 0; i < A.dim(); i++)
    {
        for (size_t j = 0; j < A.dim(); j++)
        {
            Ar[i] += A[i](j) * r[j];
        }
    }

    u = scalar_product(r, Ar, A.dim()) / scalar_product(r, r, A.dim());

    delete[] x_prev;

    return make_pair(r, u);
}


double* solve(Matrix A)
{
    double* x = new double[A.dim()];
    double u;
    pair<double*, double> tmp;
    double mu = 0;

    for (size_t i = 0; i < A.dim(); i++)
    {
        x[i] = 1.;
    }

    for (size_t i = 0; i < 10; i++)
    {
        tmp = iteration(A, x, mu);
        x = tmp.first;
        u = tmp.second;
    }

    cout << u << endl;

    return x;
}

