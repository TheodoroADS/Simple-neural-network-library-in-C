#pragma once
#include <stddef.h>

#define EPSILON 0.0000001f
#define CLIP_MIN EPSILON
#define CLIP_MAX 1.0f - EPSILON

typedef float (*Loss_func)(size_t, float*, float*);

typedef float (*Loss_derivative)(float, float);

void clip(size_t size, float* layer_values, float min, float max);

float mean_square_error(size_t size, float* layer_values, float* reference_values);

float d_mean_square_error(float layer_value, float reference_value);

float categorical_cross_entropy(size_t size, float* layer_values, float* reference_values);

float d_categorical_cross_entropy(float layer_value, float reference_value);

Loss_derivative get_loss_derivative(Loss_func lossFunc);