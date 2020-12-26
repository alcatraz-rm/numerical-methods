#define _CRT_SECURE_NO_WARNINGS

#include <stdio.h>
#include <stdlib.h>

#include "read_matrix.h"
#include "tridiagonal_method.h"

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
	//if (height != width - 1)
	//{
	//	printf("Can't solve this.\n");
	//	return 0;
	//}
	//
	//double **matrix = read_matrix(height, width, input);
	//fclose(input);

	//if (!matrix) {
	//	printf("Can't read matrix.\n");
	//	return 1;
	//}

	int height = 100;
	double** matrix = generate_matrix_exp(height);
	double *solutions = tridiagonal_matrix_algorithm(matrix, height);

	printf("Solutions:\n");

	for (size_t i = 0; i < height; i++)
	{
		printf("x%d = %lf\n", i + 1, solutions[i]);
	}

	free_matrix(matrix, height);
	delete[] solutions;

	return 0;
}