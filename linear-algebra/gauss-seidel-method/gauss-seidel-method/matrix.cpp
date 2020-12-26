#include <cstdio>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <vector>
#include <functional>
#include <stdexcept>

#include "matrix.h"

using namespace std;


double** allocate_memory_for_matrix(size_t dim)
{
	double** matrix = new double* [dim];

	for (size_t i = 0; i < dim; i++)
	{
		matrix[i] = new double[dim];
	}

	return matrix;
}

void copy_matrix(double** dest, double** source, size_t dim)
{
	for (size_t i = 0; i < dim; i++)
	{
		for (size_t j = 0; j < dim; j++)
		{
			dest[i][j] = source[i][j];
		}
	}
}

void delete_matrix(double** source, size_t dim)
{
	for (size_t i = 0; i < dim; i++)
	{
		delete[] source[i];
	}
	delete[] source;
}

Matrix::Matrix() : dimension(0), content(nullptr) {}

Matrix::Matrix(const size_t dim, double* diag_elements) : dimension(dim), content(allocate_memory_for_matrix(dim))
{
	for (size_t i = 0; i < dim; i++)
	{
		for (size_t j = 0; j < dim; j++)
		{
			if (i == j)
			{
				this->content[i][j] = diag_elements[i];
			}
			else
			{
				this->content[i][j] = 0.0;
			}
		}
	}
}

// Initializing I here
Matrix::Matrix(const size_t dim) : dimension(dim), content(allocate_memory_for_matrix(dim))
{
	for (size_t i = 0; i < dim; i++)
	{
		for (size_t j = 0; j < dim; j++)
		{
			this->content[i][j] = (i == j);
		}
	}
}

Matrix::Matrix(const Matrix& that) : dimension(that.dimension), content(allocate_memory_for_matrix(that.dimension))
{
	copy_matrix(this->content, that.content, this->dimension);
}

Matrix::Matrix(const size_t dim, ifstream& in) : dimension(dim), content(allocate_memory_for_matrix(dim))
{
	for (size_t i = 0; i < this->dimension; i++)
	{
		for (size_t j = 0; j < this->dimension; j++)
		{
			in >> this->content[i][j];
		}
	}
}

// matrix dimension getter
size_t Matrix::dim() const { return this->dimension; }

Matrix& Matrix::operator=(const Matrix& that)
{
	if (*this != that)
	{
		delete_matrix(this->content, this->dimension);

		this->dimension = that.dimension;
		this->content = allocate_memory_for_matrix(this->dimension);

		copy_matrix(this->content, that.content, this->dimension);
	}

	return *this;
}

// matrix product
Matrix Matrix::operator*(const Matrix& that) const
{
	if (that.dimension != this->dimension)
	{
		throw invalid_argument("Can't multiple matrixes with different dimensions.");
	}

	Matrix result(this->dimension);

	for (size_t i = 0; i < this->dimension; i++)
	{
		for (size_t j = 0; j < this->dimension; j++)
		{
			result.content[i][j] = 0;

			for (size_t k = 0; k < this->dimension; k++)
			{
				result.content[i][j] += this->content[i][k] * that.content[k][j];
			}
		}
	}

	return result;
}

Matrix Matrix::operator*(const double constant) const
{
	Matrix result(*this);

	for (size_t i = 0; i < this->dimension; i++)
	{
		for (size_t j = 0; j < this->dimension; j++)
		{
			result.content[i][j] *= constant;
		}
	}

	return result;
}

//matrix sum
Matrix Matrix::operator+(const Matrix& that) const
{
	if (that.dimension != this->dimension)
	{
		throw invalid_argument("Can't add matrixes with different dimensions.");
	}

	Matrix result(this->dimension);

	for (size_t i = 0; i < this->dimension; i++)
	{
		for (size_t j = 0; j < this->dimension; j++)
		{
			result.content[i][j] = this->content[i][j] + that.content[i][j];
		}
	}

	return result;
}

// adds row j to row i
void Matrix::sum_rows(size_t i, size_t j)
{
	if (i >= this->dimension || j >= this->dimension)
	{
		throw invalid_argument("Index is greater than dimension");
	}

	for (size_t k_1 = 0; k_1 < this->dimension; k_1++)
	{
		this->content[i][k_1] += this->content[j][k_1];
	}
}

void Matrix::swap_rows(size_t i, size_t j)
{
	if (i >= this->dimension || j >= this->dimension)
	{
		throw invalid_argument("Index is greater than dimension");
	}

	double* buffer = this->content[i];
	this->content[i] = this->content[j];
	this->content[j] = buffer;
}

Matrix Matrix::operator-(const Matrix& that) const
{
	Matrix tmp = that;
	tmp = tmp * (-1);

	return tmp + *this;
}

Matrix Matrix::operator-()
{
	return *this * (-1);
}

Matrix& Matrix::operator+=(const Matrix& that)
{
	*this = *this + that;

	return *this;
}

Matrix& Matrix::operator-=(const Matrix& that)
{
	*this = *this - that;

	return *this;
}

Matrix& Matrix::operator*=(const Matrix& that)
{
	*this = *this * that;

	return *this;
}

Matrix& Matrix::operator*=(const double constant)
{
	*this = *this * constant;

	return *this;
}

//matrix equality
bool Matrix::operator==(const Matrix& that) const
{
	if (this->dimension != that.dimension)
	{
		return false;
	}

	for (size_t i = 0; i < this->dimension; i++)
	{
		for (size_t j = 0; j < this->dimension; j++)
		{
			if (this->content[i][j] != that.content[i][j])
			{
				return false;
			}
		}
	}

	return true;
}

bool Matrix::operator!=(const Matrix& that) const
{
	return !(*this == that);
}

// transponse matrix
Matrix Matrix::operator~() const
{
	Matrix result(*this);

	for (size_t i = 0; i < this->dimension; i++)
	{
		for (size_t j = i + 1; j < this->dimension; j++)
		{
			if (i != j)
			{
				swap(result.content[i][j], result.content[j][i]);
			}
		}
	}

	return result;
}

Matrix Matrix::operator()(size_t row, size_t column) const
{
	Matrix result(this->dimension - 1);

	for (size_t i = 0; i < this->dimension - 1; i++)
	{
		for (size_t j = 0; j < this->dimension - 1; j++)
		{
			if (i < row && j < column)
			{
				result.content[i][j] = this->content[i][j];
			}
			else if (i < row && j >= column)
			{
				result.content[i][j] = this->content[i][j + 1];
			}
			else if (i >= row && j < column)
			{
				result.content[i][j] = this->content[i + 1][j];
			}
			else
			{
				result.content[i][j] = this->content[i + 1][j + 1];
			}
		}
	}

	return result;
}

Row Matrix::operator[](const size_t index) {
	if (index >= this->dimension)
	{
		throw invalid_argument("Row index is greater than dimension.");
	}

	return Row(index, *this);
}

Column Matrix::operator()(const size_t index) {
	if (index >= this->dimension)
	{
		throw invalid_argument("Row index is greater than dimension.");
	}

	return Column(index, *this);
}

void Matrix::print() const
{
	cout << endl;

	for (size_t i = 0; i < this->dimension; i++)
	{
		for (size_t j = 0; j < this->dimension; j++)
		{
			cout << this->content[i][j] << " ";
		}
		cout << endl;
	}
}

void Matrix::print(ofstream& out) const
{
	for (size_t i = 0; i < this->dimension; i++)
	{
		for (size_t j = 0; j < this->dimension; j++)
		{
			out << this->content[i][j] << " ";
		}
		out << endl;
	}
}

ostream& operator<<(ostream& out, Matrix matrix)
{
	for (size_t i = 0; i < matrix.dimension; i++)
	{
		for (size_t j = 0; j < matrix.dimension; j++)
		{
			out << matrix.content[i][j] << " ";
		}
		out << endl;
	}

	return out;
}

istream& operator>>(istream& in, Matrix& matrix)
{
	for (size_t i = 0; i < matrix.dimension; i++)
	{
		for (size_t j = 0; j < matrix.dimension; j++)
		{
			in >> matrix.content[i][j];
		}
	}

	return in;
}


Matrix::~Matrix()
{
	delete_matrix(this->content, this->dimension);
}


Matrix operator*(double constant, const Matrix& A)
{
	return A * constant;
}


Row::Row(size_t idx, Matrix& M) : row_index(idx), matrix(M) {}

double& Row::operator[](size_t idx)
{
	if (idx >= this->matrix.dimension)
	{
		throw invalid_argument("Row index is greater than dimension.");
	}

	return this->matrix.content[this->row_index][idx];
}

double Row::operator()(size_t idx)
{
	if (idx >= this->matrix.dimension)
	{
		throw invalid_argument("Row index is greater than dimension.");
	}

	return this->matrix.content[this->row_index][idx];
}

Column::Column(size_t idx, Matrix& M) : column_index(idx), matrix(M) {}

double& Column::operator[](size_t idx)
{
	if (idx >= this->matrix.dimension)
	{
		throw invalid_argument("Column index is greater than dimension.");
	}

	return this->matrix.content[idx][this->column_index];
}

double Column::operator()(size_t idx)
{
	if (idx >= this->matrix.dimension)
	{
		throw invalid_argument("Column index is greater than dimension.");
	}

	return this->matrix.content[idx][this->column_index];
}