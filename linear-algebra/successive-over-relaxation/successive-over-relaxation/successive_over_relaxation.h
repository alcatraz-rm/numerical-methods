#pragma once
#include "matrix.h"

double* iteration(Matrix M, Matrix inversed, double* x_prev, double* b);

double* solve(Matrix M, double* b);
