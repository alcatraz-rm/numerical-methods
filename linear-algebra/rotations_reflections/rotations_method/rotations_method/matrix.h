#pragma once

#include <fstream>
using namespace std;


double** allocate_memory_for_matrix(size_t dim);

void copy_matrix(double** dest, double** source, size_t dim);

void delete_matrix(double** source, size_t dim);

class Matrix;

class row {
	const size_t row_index;
	Matrix& matrix;

public:
	row(size_t idx, Matrix& M);
	double operator()(size_t idx);
	double& operator[](size_t idx);
};


class column {
	const size_t column_index;
	Matrix& matrix;

public:
	column(size_t idx, Matrix& M);

	double& operator[](size_t idx);
};


class Matrix {
	double** content;
	size_t dimension;

	friend class row;
	friend class column;

public:
	Matrix();

	Matrix(const size_t dim, double* diag_elements);

	// Initializing I here
	Matrix(const size_t dim);

	Matrix(const Matrix& that);

	Matrix(const size_t dim, ifstream& in);

	// matrix dimension getter
	size_t dim() const;

	void swap_rows(size_t i, size_t j);

	Matrix& operator=(const Matrix& that);

	// matrix product
	Matrix operator*(const Matrix& that) const;

	Matrix operator*(const double constant) const;

	//matrix sum
	Matrix operator+(const Matrix& that) const;

	Matrix operator-(const Matrix& that) const;

	Matrix operator-();

	void operator+=(const Matrix& that);

	void operator-=(const Matrix& that);

	void operator*=(const Matrix& that);

	void operator*=(const double constant);

	//matrix equality
	bool operator==(const Matrix& that) const;

	bool operator!=(const Matrix& that) const;

	// transponse matrix
	Matrix operator~() const;

	Matrix operator()(size_t row, size_t column) const;

	row operator[](const size_t index);

	column operator()(const size_t index);

	void print() const;

	void print(string filename) const;

	~Matrix();

};


Matrix operator*(double constant, const Matrix& A);
