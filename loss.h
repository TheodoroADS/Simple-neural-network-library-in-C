#pragma once
#include <stddef.h>

#define CLIP_MIN 0.00001
#define CLIP_MAX 0.99999

typedef double (*Loss_func)(size_t, double*, double*);

typedef double (*Loss_derivative)(double, double);

void clip(size_t size, double* layer_values, double min, double max);

double mean_square_error(size_t size, double* layer_values, double* reference_values);

double d_mean_square_error(double layer_value, double reference_value);

double categorical_cross_entropy(size_t size, double* layer_values, double* reference_values);

double d_categorical_cross_entropy(double layer_value, double reference_value);

Loss_derivative get_loss_derivative(Loss_func lossFunc);