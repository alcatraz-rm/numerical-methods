#pragma once

#include <stdio.h>

double** allocate_matrix(size_t height, size_t width);

void free_matrix(double** matrix, size_t height);

double** read_matrix(size_t height, size_t width, FILE* input);

double** generate_matrix_lemer(size_t height);

double** generate_E(size_t height);
