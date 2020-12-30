#include <iostream>
#include <fstream>
using namespace std;

#include "power_iteration.h"
#include "matrix.h"


int main()
{
	ifstream input("input.txt");
	size_t dim;
	input >> dim;
	Matrix A(dim);

	input >> A;
	input.close();

	double* r = find_max_eigenvector(A);

	for (size_t i = 0; i < A.dim(); i++)
	{
		cout << r[i] << endl;
	}
	delete[] r;

	return EXIT_SUCCESS;
}
