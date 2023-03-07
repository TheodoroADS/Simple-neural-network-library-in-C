#pragma once

#include "matrix.h"
#include "activation.h"
#include "loss.h"
#include "optimizer.h"

typedef struct Hidden_layer{

    Matrix weights;
    Matrix biases;
    Matrix values;
    Activation_func activation;
    Activation_derivative d_activation;

} Hidden_layer;

void Hidden_layer_free(Hidden_layer* layer);



typedef struct NN{

    Matrix* inputs;
    Matrix* outputs;
    size_t batch_size;
    Hidden_layer** hidden_layers;
    Matrix* output_layer_weights; 
    size_t hidden_layer_count;
    size_t allocated_layers;
    Output_Activation_func output_activation;
    Loss_func loss_function;
    Optimizer optimizer;
    int ready;


} NN;

NN* NN_create(unsigned int input_size,unsigned int batch_size,unsigned int output_size, void (*output_activation)(size_t ,float*), float (*loss_function)(size_t, float*, float*));

void NN_free(NN* network);

int NN_add_hidden(NN* network ,int layer_size, void (*layer_activation)(float*));

int NN_compile(NN* network);

// void NN_feed_foward(NN* network, Matrix* input);

float NN_eval_loss(NN* network, float** reference_vals);


void NN_fit_classification(NN* network,size_t nb_examples ,size_t nb_epochs ,float** values, int* labels, float learning_rate);

int* NN_predict_class_batch(NN* network, Matrix* input, int* predictions);

int NN_predict_class(NN* network, float* X);

int* NN_predict_class_all(NN* network, size_t how_many, float** values);


