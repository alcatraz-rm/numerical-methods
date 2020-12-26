#define _CRT_SECURE_NO_WARNINGS

#include <stdio.h>
#include <stdlib.h>

void swap_rows(double** matrix, size_t n, size_t m) {
	double* buffer = matrix[n];
	matrix[n] = matrix[m];
	matrix[m] = buffer;
}

int search_nonzero(double** matrix, size_t height, size_t n) {
	for (size_t i = n + 1; i < height; i++) {
		if (matrix[i][n]) {
			swap_rows(matrix, n, i);
			return 0;
		}
	}

	return 1;
}

void multiple_row_by_number(double** matrix, size_t width, size_t row, double num) {
	for (int i = 0; i < width; i++) {
		matrix[row][i] *= num;
	}
}

// прибавляет строку с номером n к строке с номером m
void sum_rows(double** matrix, size_t width, size_t n, size_t m) {
	for (size_t i = 0; i < width; i++) {
		matrix[n][i] += matrix[m][i];
	}
}

double** make_upper_triangle(double** matrix, size_t height, size_t width) {
	for (size_t i = 0; i < height; i++) {
		if (!matrix[i][i] && search_nonzero(matrix, height, i)) {
			continue;
		}

		for (size_t j = i + 1; j < width - 1; j++) {
			if (!matrix[j][i]) {
				continue;
			}

			multiple_row_by_number(matrix, width, i, -matrix[j][i] / matrix[i][i]);
			sum_rows(matrix, width, j, i);
		}
	}

	return matrix;
}


double* express_variables(double** matrix, size_t height, size_t width)
{
	double* solution = (double*)malloc((width - 1) * sizeof(double));
	double value;

	if (!solution)
	{
		printf("Can't allocate memory for solution\n");
		return NULL;
	}

	// guess that height == width - 1

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
