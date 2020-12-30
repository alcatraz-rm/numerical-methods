#include <iostream>
#include <fstream>

#include "matrix.h"
#include "qr.h"


Matrix generate_matrix_exp(size_t height) {
	double alpha = 10;

	Matrix matrix(height);

	for (int i = 0; i < height; i++) {
		for (int j = 0; j < height; j++) {
			matrix[i][j] = exp(-alpha * abs((i - j) * (i - j)));

		}
	}

	return matrix;
}

double* generate_matrix_exp_b(Matrix m) {

	double* b = new double[m.dim()];
	for (int i = 0; i < m.dim(); i++) {
		b[i] = 0;
		for (int j = 0; j < m.dim(); j++) {
			b[i] += m[i](j);
		}
	}

	return b;
}


int main()
{
	ifstream input("input.txt");
	size_t dim;
	input >> dim;
	Matrix A(dim);
	input >> A;

	input.close();

	solve(A);

	return EXIT_SUCCESS;
}
