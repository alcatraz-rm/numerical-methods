#include <cmath>
#include <iostream>
#include "read_matrix.h"

using namespace std;

double** represent_as_product(double** A, size_t height)
{
	double** T = allocate_matrix(height, height);

	if (A[0][0] < 0)
	{
		multiple_row_by_number(A, height + 1, 0, -1);
	}

	T[0][0] = sqrt(A[0][0]);

	for (size_t i = 1; i < height; i++)
	{
		T[0][i] = A[0][i] / T[0][0];
	}

	for (size_t i = 1; i < height; i++)
	{
		for (size_t j = i; j < height; j++)
		{
			double tmp = 0.0;

			if (i == j)
			{
				for (size_t k = 0; k < i; k++)
				{
					tmp += T[k][i] * T[k][i];
				}
				T[i][i] = sqrt(A[i][i] - tmp);
				cout << A[i][i] - tmp << endl;
			}
			else
			{
				for (size_t k = 0; k < i; k++)
				{
					tmp += T[k][i] * T[k][j];
				}
				T[i][j] = (A[i][j] - tmp) / T[i][i];
			}
		}
	}

	for (size_t i = 0; i < height; i++)
	{
		for (size_t j = 0; j < height; j++)
		{
			if (i > j)
			{
				T[i][j] = 0;
			}
		}
	}

	for (size_t i = 0; i < height; i++)
	{
		for (size_t j = 0; j < height; j++)
		{
			printf("%lf ", T[i][j]);
		}

		printf("\n");
	}
	printf("\n");


	return T;
}

double** product(double** A, double** B, size_t height)
{
	double** result = allocate_matrix(height, height);

	for (size_t i = 0; i < height; i++)
	{
		for (size_t j = 0; j < height; j++)
		{
			result[i][j] = 0;
			for (size_t k = 0; k < height; k++)
			{
				result[i][j] += A[i][k] * B[k][j];
			}
		}
	}

	return result;
}


double** transponse(double** A, size_t height)
{
	double** result = allocate_matrix(height, height);

	for (size_t i = 0; i < height; i++)
	{
		for (size_t j = 0; j < height; j++)
		{
			if (i != j)
			{
				result[i][j] = A[j][i];
			}
			else
			{
				result[i][j] = A[i][j];
			}
		}
	}

	return result;
}


double** make_system(double** A, double* b, size_t height)
{
	double** result = allocate_matrix(height, height + 1);

	for (size_t i = 0; i < height; i++)
	{
		for (size_t j = 0; j < height; j++)
		{
			result[i][j] = A[i][j];
		}
		result[i][height] = b[i];
	}

	return result;
}


double* express_variables_upper(double** matrix, size_t height, size_t width)
{
	double* solution = (double*)malloc((width - 1) * sizeof(double));
	double value;

	if (!solution)
	{
		printf("Can't allocate memory for solution\n");
		return NULL;
	}

	for (int i = height - 1; i >= 0; i--)
	{
		if (matrix[i][i] != 0)
		{
			value = matrix[i][width - 1];

			for (size_t j = i + 1; j < width - 1; j++)
			{
				value -= matrix[i][j] * solution[j];
			}

			solution[i] = value / matrix[i][i];
		}
		else
		{
			printf("Can't solve (more than 1 solution or 0).\n");
			free(solution);

			return NULL;
		}

	}

	return solution;
}


double* express_variables_lower(double** matrix, size_t height, size_t width)
{
	double* solution = (double*)malloc((width - 1) * sizeof(double));
	double value;

	if (!solution)
	{
		printf("Can't allocate memory for solution\n");
		return NULL;
	}

	for (int i = 0; i < height; i++)
	{
		if (matrix[i][i] != 0)
		{
			value = matrix[i][width - 1];

			for (size_t j = 0; j < i; j++)
			{
				value -= matrix[i][j] * solution[j];
			}

			solution[i] = value / matrix[i][i];
		}
		else
		{
			printf("Can't solve (more than 1 solution or 0).\n");
			free(solution);

			return NULL;
		}

	}

	return solution;
}


double* solve(double** A, double** T, size_t height)
{
	for (size_t i = 0; i < height; i++)
	{
		for (size_t j = 0; j < height; j++)
		{
			printf("%lf ", A[i][j]);
		}

		printf("\n");
	}
	printf("\n");

	for (size_t i = 0; i < height; i++)
	{
		for (size_t j = 0; j < height; j++)
		{
			printf("%lf ", T[i][j]);
		}

		printf("\n");
	}
	printf("\n");

	double* b = new double[height];
	for (size_t i = 0; i < height; i++)
	{
		b[i] = A[i][height];
	}


	double** T_1 = transponse(T, height);

	double** T_2 = product(T_1, T, height);
	for (size_t i = 0; i < height; i++)
	{
		for (size_t j = 0; j < height; j++)
		{
			printf("%lf ", T_2[i][j]);
		}

		printf("\n");
	}
	printf("\n");
	free_matrix(T_2, height);

	double** tmp = T_1;

	T_1 = make_system(T_1, b, height);
	free_matrix(tmp, height);


	double* y = express_variables_lower(T_1, height, height + 1);

	tmp = T;
	T = make_system(T, y, height);
	free_matrix(tmp, height);

	double* x = express_variables_upper(T, height, height + 1);

	free(y);
	free_matrix(T, height);
	free_matrix(T_1, height);
	delete[] b;

	return x;
}
