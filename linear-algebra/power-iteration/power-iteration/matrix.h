#pragma once

#include <fstream>
using namespace std;


double** allocate_memory_for_matrix(size_t dim);

void copy_matrix(double** dest, double** source, size_t dim);

void delete_matrix(double** source, size_t dim);

class Matrix;

class Row {
	const size_t row_index;
	Matrix& matrix;

public:
	Row(size_t idx, Matrix& M);

	double& operator[](size_t idx);
	double operator()(size_t idx);
};


class Column {
	const size_t column_index;
	Matrix& matrix;

public:
	Column(size_t idx, Matrix& M);

	double& operator[](size_t idx);
	double operator()(size_t idx);
};


class Matrix {
	double** content;
	size_t dimension;

	friend class Row;
	friend class Column;

	friend ostream& operator<<(ostream& out, Matrix matrix);
	friend istream& operator>>(istream& in, Matrix& matrix);

public:
	Matrix();

	Matrix(const size_t dim, double* diag_elements);

	// Initializing I here
	Matrix(const size_t dim);

	Matrix(const Matrix& that);

	Matrix(const size_t dim, ifstream& in);

	// matrix dimension getter
	size_t dim() const;

	Matrix& operator=(const Matrix& that);

	// matrix product
	Matrix operator*(const Matrix& that) const;

	Matrix operator*(const double constant) const;

	//matrix sum
	Matrix operator+(const Matrix& that) const;

	Matrix operator-(const Matrix& that) const;

	Matrix operator-();

	Matrix& operator+=(const Matrix& that);

	Matrix& operator-=(const Matrix& that);

	Matrix& operator*=(const Matrix& that);

	Matrix& operator*=(const double constant);

	//matrix equality
	bool operator==(const Matrix& that) const;

	bool operator!=(const Matrix& that) const;

	// transponse matrix
	Matrix operator~() const;

	Matrix operator()(size_t row, size_t column) const;

	Row operator[](const size_t index);

	Column operator()(const size_t index);

	void print() const;

	void print(ofstream& out) const;

	~Matrix();

};


Matrix operator*(double constant, const Matrix& A);
