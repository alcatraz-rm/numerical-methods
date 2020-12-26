#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>

#include "read_matrix.h"
#include "elimination.h"

int main() {

	FILE* input = fopen("input.txt", "r");
	if (!input) {
		printf("Can't open file\n");
		return NULL;
	}

	size_t height, width;
	fscanf(input, "%zd %zd", &height, &width);

	double** matrix = read_matrix(height, width, input);
	fclose(input);

	if (!matrix) {
		printf("Can't read matrix.\n");
		return 1;
	}

	//size_t height = 5;
	//size_t width = height;

	//double** matrix = generate_matrix_lemer(height);

	//double det = determinant(matrix, height, height);
	//if (det == 0.0)
	//{
	//	printf("Det = 0.\n");
	//	return 1;
	//}

	double** inversed = inverse(matrix, height, width);
	if (inversed)
	{
		for (size_t i = 0; i < height; i++) {
			for (size_t j = 0; j < width; j++) {
				printf("%lf ", inversed[i][j]);
			}
			printf("\n");
		}

		free_matrix(inversed, height);

	}

	free_matrix(matrix, height);

	return 0;
}