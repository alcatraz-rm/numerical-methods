#include <cmath>
#include <iostream>

using namespace std;

#include "matrix.h"

bool is_valid(Matrix A)
{
	for (size_t i = 1; i < A.dim(); i++)
	{
		if (A[i](i) == 0)
		{
			return false;
		}
	}

	return true;
}

double* express_variables(Matrix A, double* b)
{
	double* solution = new double[A.dim()];

	while (!is_valid(A))
	{
		for (size_t i = 1; i < A.dim(); i++)
		{
			if (A[i](i) == 0)
			{
				for (size_t j = i + 1; j <= A.dim(); j++)
				{
					if (A[j](i) != 0)
					{
						A.swap_rows(i, j);
						swap(b[i - 1], b[j - 1]);
						break;
					}
				}
			}
		}
	}

	A.print();
	cout << endl;

	for (size_t i = A.dim(); i >= 1; i--)
	{
		double value = b[i - 1];

		for (size_t j = i + 1; j <= A.dim(); j++)
		{
			value -= A[i][j] * solution[j - 1];
		}

		solution[i - 1] = value / A[i][i];

	}

	return solution;
}

double* rotate(Matrix A, double* b)
{
	double* tmp_1 = new double[A.dim()], * tmp_2 = new double[A.dim()];
	double c, s;

	for (size_t k = 1; k < A.dim(); k++)
	{
		for (size_t i = k; i < A.dim(); i++)
		{
			c = A[k](k) / sqrt(A[k](k) * A[k](k) + A[i + 1](k) * A[i + 1](k));
			s = A[i + 1](k) / sqrt(A[k](k) * A[k](k) + A[i + 1](k) * A[i + 1](k));

			for (size_t j = 0; j < A.dim(); j++)
			{
				tmp_1[j] = A[k](j + 1) * c + A[i + 1](j + 1) * s;
				tmp_2[j] = A[k](j + 1) * (-s) + A[i + 1](j + 1) * c;
			}

			double d = b[k - 1];
			b[k - 1] = b[k - 1] * c + b[i] * s;
			b[i] = d * (-s) + b[i] * c;

			for (size_t j = 0; j < A.dim(); j++)
			{
				A[k][j + 1] = tmp_1[j];
				A[i + 1][j + 1] = tmp_2[j];

			}

			//A.print();
		}
	}

	A.print();

	delete[] tmp_1;
	delete[] tmp_2;

	return express_variables(A, b);
}


