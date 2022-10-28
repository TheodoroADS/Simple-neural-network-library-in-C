#pragma once

#include <stddef.h>

typedef void (*Activation_func)(float*); 

typedef void (*Output_Activation_func)(size_t , float*);

typedef float (*Output_Activation_derivative)(float);

typedef float (*Activation_derivative)(float);

void relu(float* x);

void leaky_relu(float* x);

float d_relu(float x);

float d_leaky_relu(float x);

void sigmoid(float* x);

float d_sigmoid(float x);

void softmax(size_t nb_values , float* values);

float d_softmax(float x);

void sigmoid_output(size_t nb_values, float* values);

Activation_derivative resolve_derivative(Activation_func func);

Output_Activation_derivative resolve_out_derivative(Output_Activation_func func);
