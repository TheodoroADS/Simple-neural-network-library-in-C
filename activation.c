#include <math.h>
#include "activation.h"
#include <stdio.h>
#include "loss.h"


void relu(double* x){
    if(*x < 0){
        *x = 0;
    }
}

double d_relu(double x){
   return  x < 0 ? 0 : 1;
}

static inline double sigmoid_func(double x){
    return 1/(1+ exp(-x));
}   

void sigmoid(double* x){
    *x = sigmoid_func(*x);
}

double d_sigmoid(double x){
    // double sigm = sigmoid_func(x);

    //return sigm*(1-sigm);
    return x*(1.0-x);
}


void softmax(size_t nb_values , double* values){

    double exp_sum = 0;

    clip(nb_values, values, CLIP_MIN, CLIP_MAX);

    for (size_t i = 0; i < nb_values; i++)
    {   
        values[i] = exp(values[i]);
        exp_sum += values[i];
    }
    
    for (size_t i = 0; i < nb_values; i++)
    {
        values[i] /= exp_sum;
    }


}


Activation_derivative resolve_derivative(Activation_func func){

    // unforutnately I cannot do a switch statement :(

    if (func == relu){
        return d_relu;
    } else if (func == sigmoid){
        return d_sigmoid;
    } else{
        fprintf(stderr, "No derivative known for function provided \n");
        return NULL;
    }

}
