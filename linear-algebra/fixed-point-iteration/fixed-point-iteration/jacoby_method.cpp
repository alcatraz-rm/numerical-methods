#include <iostream>
#include <cmath>

#include "matrix.h"
using namespace std;


pair<int, int> find_nonzero(Matrix A)
{
	double m = A[0](1);
	size_t p = 0, q = 1;

	for (size_t i = 0; i < A.dim(); i++)
	{
		for (size_t j = 0; j < A.dim(); j++)
		{
			if (i < j)
			{
				if (fabs(A[i](j)) > fabs(A[p](q)))
				{
					p = i;
					q = j;
				}
			}
		}
	}

	if (A[p](q) != 0)
	{
		return make_pair((int)p, (int)q);
	}

	return make_pair(-1, -1);
}


double sgn(double value)
{
	if (value < 0)
	{
		return -1;
	}
	if (value > 0)
	{
		return 1;
	}

	return 0;
}


pair<double, double> calculate_sin_cos(Matrix A, size_t p, size_t q)
{
	double tau = (A[p](p) - A[q](q)) / (2 * A[p][q]);
	double t;

	if (tau == 0)
	{
		t = 1;
	}
	else
	{
		t = -tau + sgn(tau) * sqrt(tau * tau + 1);
	}

	double c = 1 / sqrt(t * t + 1);
	double s = t * c;

	return make_pair(c, s);
}


Matrix build_G(size_t dim, pair<double, double> c_s, pair<size_t, size_t> place)
{
	Matrix result(dim);

	result[place.first][place.first] = c_s.first;
	result[place.first][place.second] = -c_s.second;
	result[place.second][place.first] = c_s.second;
	result[place.second][place.second] = c_s.first;

	return result;
}


Matrix turn_zero(Matrix A)
{
	double epsilon = 0.000001;

	for (size_t i = 0; i < A.dim(); i++)
	{
		for (size_t j = 0; j < A.dim(); j++)
		{
			if (fabs(A[i][j]) < epsilon)
			{
				A[i][j] = 0;
			}
		}
	}

	return A;
}


bool check(Matrix A)
{
	double epsilon = 0.000001;

	for (size_t i = 0; i < A.dim(); i++)
	{
		for (size_t j = 0; j < A.dim(); j++)
		{
			if ((i < j) && (fabs(A[i](j)) > epsilon))
			{
				return false;
			}
		}
	}

	return true;
}


double find_max_eigenvalue(Matrix A)
{
	while (true)
	{
		pair<int, int> indexes = find_nonzero(A);
		if (indexes.first == -1)
		{
			break;
		}
		if (check(A))
		{
			break;
			check(A);
		}

		size_t p = (size_t)indexes.first;
		size_t q = (size_t)indexes.second;

		pair<double, double> c_s = calculate_sin_cos(A, p, q);

		Matrix G;
		G = build_G(A.dim(), c_s, make_pair(p, q));

		A = (~G) * A * G;

	}

	A = turn_zero(A);

	double max_eigenvalue = A[0](0);
	//double min_eigenvalue = A[0](0);

	for (size_t i = 1; i < A.dim(); i++)
	{
		if (fabs(A[i](i)) > fabs(max_eigenvalue))
		{
			max_eigenvalue = A[i](i);
		}

	}

	return max_eigenvalue;
}
