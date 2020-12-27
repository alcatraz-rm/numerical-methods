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

    return result;
}


double* iteration(Matrix M, Matrix inversed, double* x_prev, double* b)
{
    double w = 1.5;
    double* Ax_prev = multiply_b(M, x_prev);
    double* x_prev_copy = new double[M.dim()];
    for (int i = 0; i < M.dim(); i++) {
        x_prev_copy[i] = x_prev[i];
    }


    for (size_t k = 0; k < M.dim(); k++)
    {
        Ax_prev[k] -= b[k];
    }

    double* tmp = multiply_b(inversed, Ax_prev);

    for (size_t i = 0; i < M.dim(); i++)
    {
        x_prev[i] -= tmp[i];
        x_prev[i] *= w;
        x_prev[i] += (1 - w) * x_prev_copy[i];
    }

    delete[] Ax_prev;
    delete[] tmp;
    delete[] x_prev_copy;

    return x_prev;
}


double* copy_vector(double* vector, size_t height)
{
    double* result = new double[height];

    for (size_t i = 0; i < height; i++)
    {
        result[i] = vector[i];
    }

    return result;
}


double* solve(Matrix M, double* b)
{
    Matrix L = M;
    Matrix D = M;

    for (size_t i = 0; i < M.dim(); i++)
    {
        for (size_t j = 0; j < M.dim(); j++)
        {
            if (i <= j)
            {
                L[i][j] = 0;
            }

            if (i != j)
            {
                D[i][j] = 0;
            }
        }
    }

    Matrix D_L_inversed;
    Matrix tmp;
    tmp = D - L;
    D_L_inversed = inverse_matrix(tmp);

    double* x = new double[M.dim()];
    for (size_t i = 0; i < M.dim(); i++)
    {
        x[i] = 1;
    }

    double epsilon = pow(10, -5);

    while (true)
    {
        double* x_copy = copy_vector(x, M.dim());
        x = iteration(M, D_L_inversed, x, b);

        for (size_t j = 0; j < M.dim(); j++)
        {
            x_copy[j] -= x[j];
        }
        double length = getLength(x_copy, M.dim());

        if (length < epsilon)
        {
            break;
        }
    }

    return x;
}
