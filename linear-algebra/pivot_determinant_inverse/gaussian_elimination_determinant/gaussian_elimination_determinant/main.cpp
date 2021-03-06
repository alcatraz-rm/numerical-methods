#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>

#include "read_matrix.h"
#include "elimination.h"

int main() {

	//FILE* input = fopen("input.txt", "r");
	//if (!input) {
	//	printf("Can't open file\n");
	//	return NULL;
	//}
	//
	//size_t height, width;
	//fscanf(input, "%zd %zd", &height, &width);
	//
	//double **matrix = read_matrix(height, width, input);
	//fclose(input);
	//
	//if (!matrix) {
	//	printf("Can't read matrix.\n");
	//	return 1;
	//}

	size_t height = 10;
	size_t width = height;

	double** matrix = generate_matrix_one(height);

	double det = calculate_determinant(matrix, height, width);

	for (size_t i = 0; i < height; i++) {
		for (size_t j = 0; j < width; j++) {
			printf("%lf ", matrix[i][j]);
		}
		printf("\n");
	}
	printf("%lf ", det);

	free_matrix(matrix, height);

	return 0;
}