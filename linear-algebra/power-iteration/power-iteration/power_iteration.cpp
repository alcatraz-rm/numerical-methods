#include <iostream>

#include "matrix.h"


double length(double* v, size_t dim)
{
	double result = 0;

	for (size_t i = 0; i < dim; i++)
	{
		result += v[i] * v[i];
	}

	return sqrt(result);
}


double scalar_product(double* one, double* two, size_t dim)
{
	double result = 0;

	for (size_t i = 0; i < dim; i++)
	{
		result += one[i] * two[i];
	}

	return result;
}


pair<double*, double> iteration(Matrix A, double* r_prev)
{
	double* r = new double[A.dim()], * Ar_copy = new double[A.dim()];
	double u;

	for (size_t i = 0; i < A.dim(); i++)
	{
		r[i] = 0;
		for (size_t j = 0; j < A.dim(); j++)
		{
			r[i] += A[i](j) * r_prev[j];
		}
	}

	for (size_t j = 0; j < A.dim(); j++)
	{
		Ar_copy[j] = r[j];
	}

	double r_length = length(r, A.dim());

	for (size_t j = 0; j < A.dim(); j++)
	{
		r[j] /= r_length;
	}


	u = scalar_product(r, Ar_copy, A.dim()) / scalar_product(r, r, A.dim());

	delete[] r_prev;
	delete[] Ar_copy;

	return make_pair(r, u);
}

double* find_max_eigenvector(Matrix A)
{
	double* r_0 = new double[A.dim()];
	for (size_t i = 0; i < A.dim(); i++)
	{
		r_0[i] = 1;
	}

	double epsilon = pow(10, -15);

	double* r, u;
	pair<double*, double> tmp = iteration(A, r_0);
	r = tmp.first;
	double r_prev_length;

	for (size_t i = 0; i < 1000; i++)
	{
		r_prev_length = length(r, A.dim());
		pair<double*, double> tmp = iteration(A, r);
		r = tmp.first;
		u = tmp.second;

	}

	cout << u << endl;

	return r;
}
