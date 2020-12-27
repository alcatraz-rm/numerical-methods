#include <iostream>
#include <algorithm>
using namespace std;

#include "matrix.h"


int search_nonzero_down(Matrix& matrix, Matrix& result, size_t n) {
	for (size_t i = n; i < matrix.dim(); i++) {
		if (matrix[i](n)) {
			if (i != n)
			{
				matrix.sum_rows(n, i);
				result.sum_rows(n, i);
			}
			return 0;
		}

	}
	return 1;
}

int search_nonzero_up(Matrix& matrix, Matrix& result, size_t n) {
	for (int i = (int)n; i >= 0; i--) {
		if (matrix[i](n)) {
			if (i != n)
			{
				matrix.sum_rows(n, i);
				result.sum_rows(n, i);
			}
			return 0;
		}
	}
	return 1;
}

void multiple_row_by_number(Matrix& matrix, size_t row, double num)
{
	for (int i = 0; i < matrix.dim(); i++) {
		matrix[row][i] *= num;
	}
}

int upper_triangle(Matrix& matrix, Matrix& result)
{
	for (size_t i = 0; i < matrix.dim(); i++) {
		if (search_nonzero_down(matrix, result, i)) {
			return -1;
		}

		for (size_t j = i + 1; j < matrix.dim(); j++) {
			if (!matrix[j](i)) {
				continue;
			}

			multiple_row_by_number(result, i, -matrix[j](i) / matrix[i](i));
			multiple_row_by_number(matrix, i, -matrix[j](i) / matrix[i](i));
			matrix.sum_rows(j, i);
			result.sum_rows(j, i);
		}
	}

	return 0;
}

int lower_triangle(Matrix& matrix, Matrix& result)
{
	int i = (int)(matrix.dim() - 1);

	while (i >= 0) {
		if (search_nonzero_up(matrix, result, i)) {
			return -1;
		}

		int j = i - 1;

		while (j >= 0) {
			if (!matrix[j](i)) {
				j--;
				continue;
			}

			multiple_row_by_number(result, i, -matrix[j](i) / matrix[i](i));
			multiple_row_by_number(matrix, i, -matrix[j](i) / matrix[i](i));
			matrix.sum_rows(j, i);
			result.sum_rows(j, i);
			j--;
		}

		i--;

	}

	return 0;
}

Matrix inverse_matrix(Matrix& matrix) {
	Matrix result(matrix.dim());

	int status = upper_triangle(matrix, result);

	if (status == -1)
	{
		cout << "Det = 0" << endl;
		throw;
	}
	status = lower_triangle(matrix, result);

	for (size_t i = 0; i < matrix.dim(); i++)
	{
		multiple_row_by_number(result, i, 1 / matrix[i](i));
		multiple_row_by_number(matrix, i, 1 / matrix[i](i));
	}

	return result;
}
