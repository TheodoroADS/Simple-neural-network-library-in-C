#pragma once

#include <stddef.h>

typedef void (*Activation_func)(double*); 

typedef void (*Output_Activation_func)(size_t , double*);

typedef double (*Activation_derivative)(double);

void relu(double* x);

double d_relu(double x);

void sigmoid(double* x);

double d_sigmoid(double x);

void softmax(size_t nb_values , double* values);

void sigmoid_output(size_t nb_values, double* values);

Activation_derivative resolve_derivative(Activation_func func);

