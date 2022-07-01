#include <stdio.h>
#include "matrix.h"
#include <math.h>
#include "loss.h"


void clip(size_t size, double* layer_values, double min, double max){

    for(size_t i = 0; i < size; i++){
        

        if (layer_values[i] >  max){
            layer_values[i] = max;
        }else if(layer_values[i] < min){
            layer_values[i] = min;
        }
    }

}

double mean_square_error(size_t size, double* layer_values, double* reference_values){
    
    double total_square_error = 0;
    double err;

    for (size_t i = 0; i < size; i++)
    {
        err = reference_values[i] - layer_values[i];
        total_square_error = err*err;
    }
    
    return total_square_error / ( (double) size);
}

double d_mean_square_error(size_t size, double* layer_values, double* reference_values, size_t neuron_index ){

    return 2*(layer_values[neuron_index] - reference_values[neuron_index]);
}

double categorical_cross_entropy(size_t size, double* layer_values, double* reference_values){


    //clipping values to avoid having ln(0) or ln(1)
    clip(size, layer_values, CLIP_MIN, CLIP_MAX);

    double loss = 0;


    for (size_t i = 0; i < size; i++)
    {
        loss -= reference_values[i]*log(layer_values[i]);
    }
    

    return loss;
}

double d_categorical_cross_entropy(size_t size,double* layer_values, double* reference_values, size_t neuron_index){
    return - reference_values[neuron_index]/layer_values[neuron_index];
}

Loss_derivative get_loss_derivative(Loss_func lossFunc){

    if (lossFunc == mean_square_error)  
    {
        return d_mean_square_error;
    }
    else if(lossFunc == categorical_cross_entropy){
    
        return d_categorical_cross_entropy;
    }else{
        fprintf(stderr, "Derivative of loss function provided is not known \n");
        return NULL;

    }

}