#pragma once
#include <stddef.h>

#define EPSILON 0.0000001
#define CLIP_MIN EPSILON
#define CLIP_MAX 1 - EPSILON

typedef double (*Loss_func)(size_t, double*, double*);

typedef double (*Loss_derivative)(double, double);

void clip(size_t size, double* layer_values, double min, double max);

double mean_square_error(size_t size, double* layer_values, double* reference_values);

double d_mean_square_error(double layer_value, double reference_value);

double categorical_cross_entropy(size_t size, double* layer_values, double* reference_values);

double d_categorical_cross_entropy(double layer_value, double reference_value);

Loss_derivative get_loss_derivative(Loss_func lossFunc);