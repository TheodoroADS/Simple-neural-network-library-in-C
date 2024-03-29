#pragma once

#include <stdlib.h>

typedef struct{

    size_t nb_rows;
    size_t nb_cols;
    float** data;

} Matrix;

Matrix* matrix_new( size_t nb_rows , size_t nb_cols);

Matrix* matrix_zeros( size_t nb_rows , size_t nb_cols);

Matrix* matrix_random( size_t nb_rows , size_t nb_cols);

Matrix* as_vector(size_t dim, float* values);

Matrix* as_batch(size_t batch_size, size_t dim , float** values, Matrix* batch);

Matrix* matrix_of(size_t nb_rows , size_t nb_cols , float value);

void matrix_delete(Matrix* mat);

void matrix_apply(Matrix* mat, void (*func)(float*));

void matrix_apply_rows(Matrix* mat,void (*func)(size_t, float*));

void matrix_render(Matrix* mat);

void matrix_mul(Matrix* A, Matrix* B, Matrix* Res);

void matrix_add(Matrix* M1, Matrix* M2);

size_t matrix_argmax(Matrix* mat, size_t axis, size_t pos);

Matrix* matrix_transposed(Matrix* mat);

