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

double** generate_matrix_lemer(size_t height) {
	double** matrix = allocate_matrix(height, height);

	for (int i = 0; i < height; i++) {
		for (int j = 0; j < height; j++) {
			if (j < i) {
				matrix[i][j] = ((j + 1) * 1.0) / ((i + 1) * 1.0);
			}
			if (j >= i) {
				matrix[i][j] = ((i + 1) * 1.0) / ((j + 1) * 1.0);
			}
		}
	}

	return matrix;
}

double** generate_E(size_t height)
{
	double** matrix = allocate_matrix(height, height);

	for (size_t i = 0; i < height; i++) {
		for (size_t j = 0; j < height; j++) {
			if (j != i) {
				matrix[i][j] = 0;
			}
			else {
				matrix[i][j] = 1.0;
			}
		}
	}

	return matrix;
}

