#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

void swap_rows(double** matrix, size_t n, size_t m) {
	double* buffer = matrix[n];
	matrix[n] = matrix[m];
	matrix[m] = buffer;
}

int search_max(double** matrix, size_t height, size_t n, double* result) {
	double max_value = matrix[n][n];
	size_t max_index = n;

	for (size_t i = n; i < height; i++) {
		if (fabs(matrix[i][n]) > max_value) {
			max_value = fabs(matrix[i][n]);
			max_index = i;
		}
	}

	if (matrix[max_index][n] != 0)
	{
		if (max_index != n)
		{
			swap_rows(matrix, n, max_index);
			*result *= (-1);
		}
		return 0;
	}

	return 1;
}

void multiple_row_by_number(double** matrix, size_t width, size_t row, double num, double* result) {
	for (int i = 0; i < width; i++) {
		matrix[row][i] *= num;
	}
	*result /= num;
}

// прибавляет строку с номером n к строке с номером m
void sum_rows(double** matrix, size_t width, size_t n, size_t m) {
	for (size_t i = 0; i < width; i++) {
		matrix[n][i] += matrix[m][i];
	}
}

double calculate_determinant(double** matrix, size_t height, size_t width) {
	double result = 1.0;

	if (height == 1) {
		return matrix[0][0];
	}

	if (height == 2) {
		return matrix[0][0] * matrix[1][1] - matrix[1][0] * matrix[0][1];
	}

	for (size_t i = 0; i < height; i++) {
		if (search_max(matrix, height, i, &result)) {
			continue;
		}

		for (size_t j = i + 1; j < width; j++) {
			if (!matrix[j][i]) {
				continue;
			}

			multiple_row_by_number(matrix, width, i, -matrix[j][i] / matrix[i][i], &result);
			sum_rows(matrix, width, j, i);
		}
	}

	for (size_t i = 0; i < height; i++)
	{
		result *= matrix[i][i];
	}

	return result;
}
