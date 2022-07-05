#pragma once

#include "matrix.h"
#include "activation.h"
#include "loss.h"


typedef struct Hidden_layer{

    Matrix weights;
    Matrix biases;
    Matrix values;
    Activation_func activation;
    Activation_derivative d_activation;

} Hidden_layer;



void foward(Hidden_layer layer,Matrix* input, Matrix* output);



typedef struct NN{

    Matrix* inputs;
    Matrix* outputs;
    int batch_size;
    Hidden_layer** hidden_layers;
    Matrix* output_layer_weights; 
    int hidden_layer_count;
    int allocated_layers;
    Output_Activation_func output_activation;
    Loss_func loss_function;
    Loss_derivative d_loss_function;
    int ready;


} NN;

NN NN_create(int input_size, int batch_size, int output_size, void (*output_activation)(size_t ,double*), double (*loss_function)(size_t, double*, double*));

int NN_add_hidden(NN* network ,int layer_size, void (*layer_activation)(double*));
int NN_compile(NN* network);

// void NN_feed_foward(NN* network, Matrix* input);

double NN_eval_loss(NN* network, double** reference_vals);


void NN_fit_classification(NN* network,size_t nb_examples ,size_t nb_epochs ,double** values, int* labels, double learning_rate);

int* NN_predict_class_batch(NN* network, Matrix* input, int* predictions);

int* NN_predict_class_all(NN* network, size_t how_many, double** values);


