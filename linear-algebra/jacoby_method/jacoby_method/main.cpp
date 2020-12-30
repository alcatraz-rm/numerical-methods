#include <iostream>
#include <fstream>

#include "matrix.h"
#include "jacoby_method.h"

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
