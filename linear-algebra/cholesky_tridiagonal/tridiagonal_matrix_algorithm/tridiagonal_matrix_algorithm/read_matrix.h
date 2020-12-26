#pragma once

#include <stdio.h>

double** allocate_matrix(size_t height, size_t width);

void free_matrix(double** matrix, size_t height);

double** read_matrix(size_t height, size_t width, FILE* input);

double** generate_matrix_exp(size_t height);

double** generate_matrix_one(size_t height);

void swap_rows(double** matrix, size_t n, size_t m);

int search_nonzero(double** matrix, size_t height, size_t n);

void multiple_row_by_number(double** matrix, size_t width, size_t row, double num);

// прибавляет строку с номером n к строке с номером m
void sum_rows(double** matrix, size_t width, size_t n, size_t m);
