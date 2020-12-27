#include "matrix.h"
#include "jacoby_method.h"


double* iteration(Matrix A, double* b, double* x_prev, double tau)
{
	double* Ax_b = new double[A.dim()];
	for (size_t i = 0; i < A.dim(); i++)
	{
		Ax_b[i] = -b[i];
		for (size_t j = 0; j < A.dim(); j++)
		{
			Ax_b[i] += A[i](j) * x_prev[j];
		}
		Ax_b[i] *= tau;
	}

	for (size_t i = 0; i < A.dim(); i++)
	{
		x_prev[i] -= Ax_b[i];
	}

	delete[] Ax_b;
	return x_prev;
}


double length(double* v, size_t dim)
{
	double result = 0;

	for (size_t i = 0; i < dim; i++)
	{
		result += v[i] * v[i];
	}

	return sqrt(result);
}


double* solve(Matrix A, double* b)
{
	double spectral_radius = fabs(find_max_eigenvalue(A));
	double tau = 1 / spectral_radius;

	double* x_0 = new double[A.dim()];
	for (size_t i = 0; i < A.dim(); i++)
	{
		x_0[i] = 1;
	}

	double epsilon = pow(10, -9);

	double* x = x_0;
	double x_prev_length;

	while (1)
	{
		x_prev_length = length(x, A.dim());
		x = iteration(A, b, x, tau);

		if (fabs(length(x, A.dim()) - x_prev_length) < epsilon)
		{
			break;
		}
	}

	return x;
}
