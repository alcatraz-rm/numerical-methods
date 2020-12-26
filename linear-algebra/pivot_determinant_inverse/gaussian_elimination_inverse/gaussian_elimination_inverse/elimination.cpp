#define _CRT_SECURE_NO_WARNINGS

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <algorithm>
using namespace std;

#include "read_matrix.h"

// прибавляет строку с номером m к строке с номером n
void sum_rows(double** matrix, size_t width, size_t n, size_t m) {
	for (size_t i = 0; i < width; i++) {
		matrix[n][i] += matrix[m][i];
	}
}

void swap_rows(double** matrix, size_t n, size_t m) {
	double* buffer = matrix[n];
	matrix[n] = matrix[m];
	matrix[m] = buffer;
}

int search_nonzero_down(double** matrix, double** result, size_t height, size_t n) {
	for (size_t i = n; i < height; i++) {
		if (matrix[i][n]) {
			if (i != n)
			{
				sum_rows(matrix, height, n, i);
				sum_rows(result, height, n, i);
			}
			return 0;
		}

		return 1;
	}
}

int search_nonzero_up(double** matrix, double** result, size_t height, size_t n) {
	for (int i = (int)n; i >= 0; i--) {
		if (matrix[i][n]) {
			if (i != n)
			{
				sum_rows(matrix, height, n, i);
				sum_rows(result, height, n, i);
			}
			return 0;
		}

		return 1;
	}
}

void multiple_row_by_number(double** matrix, size_t width, size_t row, double num)
{
	for (int i = 0; i < width; i++) {
		matrix[row][i] *= num;
	}
}

int upper_triangle(double** matrix, double** result, size_t height, size_t width)
{
	for (size_t i = 0; i < height; i++) {
		if (search_nonzero_down(matrix, result, height, i)) {
			return -1;
		}

		for (size_t j = i + 1; j < width; j++) {
			if (!matrix[j][i]) {
				continue;
			}

			multiple_row_by_number(result, width, i, -matrix[j][i] / matrix[i][i]);
			multiple_row_by_number(matrix, width, i, -matrix[j][i] / matrix[i][i]);
			sum_rows(matrix, width, j, i);
			sum_rows(result, width, j, i);
		}
	}

	return 0;
}

int lower_triangle(double** matrix, double** result, size_t height, size_t width)
{
	int i = (int)(height - 1);

	while (i >= 0) {
		if (search_nonzero_up(matrix, result, height, i)) {
			return -1;
			//continue;
		}

		int j = i - 1;

		while (j >= 0) {
			if (!matrix[j][i]) {
				j--;
				continue;
			}

			multiple_row_by_number(result, width, i, -matrix[j][i] / matrix[i][i]);
			multiple_row_by_number(matrix, width, i, -matrix[j][i] / matrix[i][i]);
			sum_rows(matrix, width, j, i);
			sum_rows(result, width, j, i);
			j--;
		}

		i--;

	}

	return 0;
}

double** inverse(double** matrix, size_t height, size_t width) {
	double** result = generate_E(height);

	int status = upper_triangle(matrix, result, height, width);
	if (status == -1)
	{
		printf("Det = 0\n");
		free_matrix(result, height);
		return NULL;
	}
	status = lower_triangle(matrix, result, height, width);

	for (size_t i = 0; i < height; i++)
	{
		multiple_row_by_number(result, width, i, 1 / matrix[i][i]);
		multiple_row_by_number(matrix, width, i, 1 / matrix[i][i]);
	}

	return result;
}
