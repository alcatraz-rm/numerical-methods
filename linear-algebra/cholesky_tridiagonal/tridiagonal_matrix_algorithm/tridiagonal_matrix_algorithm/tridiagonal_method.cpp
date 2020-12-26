#define _CRT_SECURE_NO_WARNINGS

#include <iostream>

using namespace std;

double* tridiagonal_matrix_algorithm(double** matrix, int height) {
	double* d = new double[height];
	double* c = new double[height - 1];
	double* b = new double[height];
	double* a = new double[height - 1];

	for (int i = 0; i < height; i++) {
		for (int j = 0; j < height; j++) {
			if (i == j) {
				b[i] = matrix[i][j];
			}
			if (i == j - 1) {
				c[i] = matrix[i][j];
			}
			if (j == i - 1) {
				a[j] = matrix[i][j];
			}
		}
		d[i] = matrix[i][height];
	}

	for (int i = 0; i < height; i++) {
		cout << b[i] << " ";
	}
	cout << endl;

	for (int i = 0; i < height - 1; i++) {
		cout << c[i] << " ";
	}
	cout << endl;

	for (int i = 0; i < height - 1; i++) {
		cout << a[i] << " ";
	}
	cout << endl;

	for (int i = 0; i < height; i++) {
		cout << d[i] << " ";
	}
	cout << endl;

	double* P = new double[height - 1];
	double* Q = new double[height - 1];

	P[0] = c[0] / (-b[0]);
	Q[0] = -d[0] / (-b[0]);

	for (int i = 1; i < height - 1; i++) {
		P[i] = c[i] / (-a[i - 1] * P[i - 1] - b[i]);
		Q[i] = (-d[i] + a[i - 1] * Q[i - 1]) / (-a[i - 1] * P[i - 1] - b[i]);
	}

	for (int i = 0; i < height - 1; i++) {
		cout << P[i] << " ";
	}
	cout << endl;

	for (int i = 0; i < height - 1; i++) {
		cout << Q[i] << " ";
	}
	cout << endl;

	double* solutions = new double[height];
	solutions[height - 1] = (d[height - 1] - Q[height - 2] * a[height - 2]) / (a[height - 2] * P[height - 2] + b[height - 1]);
	
	for (int i = height - 2; i >= 0; i--) {
		solutions[i] = P[i] * solutions[i + 1] + Q[i];
	}

	delete[] a;
	delete[] b;
	delete[] c;
	delete[] d;
	delete[] P;
	delete[] Q;

	return solutions;
}