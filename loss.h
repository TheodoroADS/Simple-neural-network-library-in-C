#pragma once
#include <stddef.h>

#define CLIP_MIN 0.00001
#define CLIP_MAX 0.99999

typedef double (*Loss_func)(size_t, double*, double*);

typedef double (*Loss_derivative)(size_t, double*, double*, size_t);

void clip(size_t size, double* layer_values, double min, double max);

double mean_square_error(size_t size, double* layer_values, double* reference_values);

double d_mean_square_error(size_t size, double* layer_values, double* reference_values, size_t neuron_index );

double categorical_cross_entropy(size_t size, double* layer_values, double* reference_values);

double d_categorical_cross_entropy(size_t size,double* layer_values, double* reference_values, size_t neuron_index);

Loss_derivative get_loss_derivative(Loss_func lossFunc);