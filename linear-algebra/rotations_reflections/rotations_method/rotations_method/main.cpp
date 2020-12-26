#include <iostream>
#include <iomanip>
#include <fstream>
#include <cmath>

#include "matrix.h"
#include "rotations_method.h"

using namespace std;

Matrix generate_matrix_exp(size_t height) {
	double alpha = 10;

	Matrix matrix(height);

	for (int i = 1; i <= height; i++) {
		for (int j = 1; j <= height; j++) {
			matrix[i][j] = exp(-alpha * abs(i - j));

		}
	}

	return matrix;
}

double* generate_matrix_exp_b(Matrix m) {

	double* b = new double[m.dim()];
	for (int i = 1; i <= m.dim(); i++) {
		b[i - 1] = 0;
		for (int j = 1; j <= m.dim(); j++) {
			b[i - 1] += m[i][j];
		}
	}

	return b;
}

int main()
{
	//ifstream input("input.txt");
	size_t dim = 100;
	//input >> dim;
	//Matrix A(dim, input);

	Matrix A(dim);
	A = generate_matrix_exp(dim);

	double* b = generate_matrix_exp_b(A);

	//for (size_t i = 0; i < dim; i++)
	//{
	//	input >> b[i];
	//}

	//input.close();

	double* solution = rotate(A, b);

	cout << "Solutions: " << endl;

	for (size_t i = 0; i < A.dim(); i++)
	{
		cout << "x_" << setprecision(20) << i + 1 << ": " << solution[i] << endl;
	}

	delete[] solution;

	return EXIT_SUCCESS;
}
