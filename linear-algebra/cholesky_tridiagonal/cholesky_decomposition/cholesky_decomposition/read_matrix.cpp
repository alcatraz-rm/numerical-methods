#define _CRT_SECURE_NO_WARNINGS

#include <stdio.h>
#include <stdlib.h>
#include <io.h>
#include <math.h>

#include "read_matrix.h"

double** allocate_matrix(size_t height, size_t width) {
	double** result = (double**)calloc(height, sizeof(double*));

	if (!result) {
		printf("Can't allocate memory for matrix rows.");
		return NULL;
	}

	for (size_t i = 0; i < height; i++) {
		result[i] = (double*)malloc(sizeof(double) * width);

		if (!result[i]) {
			printf("Can't allocate memory for matrix row.\n");
			free_matrix(result, height);
			return NULL;
		}
	}

	return result;
}

void free_matrix(double** matrix, size_t height) {
	for (size_t i = 0; i < height; i++) {
		if (matrix[i]) {
			free(matrix[i]);
		}
	}

	free(matrix);
}

double** read_matrix(size_t height, size_t width, FILE* input) {
	double** matrix = allocate_matrix(height, width);
	if (!matrix) {
		return NULL;
	}

	for (size_t i = 0; i < height; i++) {
		for (size_t j = 0; j < width; j++) {
			fscanf(input, "%lf", &matrix[i][j]);
		}
	}

	return matrix;
}

double** generate_matrix_exp(size_t height) {
	double alpha = 3.999;
	double accum = 0;

	double** matrix = allocate_matrix(height, height + 1);

	for (size_t i = 0; i < height; i++) {
		accum = 0;
		for (size_t j = 0; j < height; j++) {
			matrix[i][j] = exp(-alpha * (i - j) * (i - j));
			accum += matrix[i][j];
		}
		matrix[i][height] = accum;
	}

	return matrix;
}

double** generate_matrix_one(size_t height) {
	double** matrix = allocate_matrix(height, height + 1);
	for (size_t i = 0; i < height; i++) {
		for (size_t j = 0; j < height; j++) {
			if (j < i) {
				matrix[i][j] = 0;
			}
			if (j == i) {
				matrix[i][j] = 1.0;
			}
			if (j > i) {
				matrix[i][j] = -1.0;
			}
		}
		if (i == height - 1) {
			matrix[i][height] = 1.0;
		}
		else {
			matrix[i][height] = -1.0;
		}
	}
	return matrix;
}

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

// прибавляет строку с номером m к строке с номером n
void sum_rows(double** matrix, size_t width, size_t n, size_t m) {
	for (size_t i = 0; i < width; i++) {
		matrix[n][i] += matrix[m][i];
	}
}
