#pragma once

#include <stddef.h>

typedef void (*Activation_func)(double*); 

typedef void (*Output_Activation_func)(size_t , double*);

typedef double (*Output_Activation_derivative)(double);

typedef double (*Activation_derivative)(double);

void relu(double* x);

void leaky_relu(double* x);

double d_relu(double x);

double d_leaky_relu(double x);

void sigmoid(double* x);

double d_sigmoid(double x);

void softmax(size_t nb_values , double* values);

double d_softmax(double x);

void sigmoid_output(size_t nb_values, double* values);

Activation_derivative resolve_derivative(Activation_func func);

Output_Activation_derivative resolve_out_derivative(Output_Activation_func func);
