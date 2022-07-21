#include <math.h>
#include "activation.h"
#include <stdio.h>
#include "loss.h"


#define LEAKY_RELU_COEF 0.1

void relu(double* x){
    if(*x < 0){
        *x = 0;
    }
}

void leaky_relu(double* x){
    if(*x < 0){
        *x *= LEAKY_RELU_COEF;
    }
}

double d_relu(double x){
   if (x < 0){
       return 0.0;
   }else{
       return 1.0;
   }
}

double d_leaky_relu(double x){
    if(x <= 0){
        return LEAKY_RELU_COEF;
    }else{
        return 1.0;
    }
}

static inline double sigmoid_func(double x){
    return 1/(1+ exp(-x));
}   

void sigmoid(double* x){
    *x = sigmoid_func(*x);
}


void sigmoid_output(size_t nb_values, double* values){
    for (size_t i = 0; i < nb_values; i++)
    {
        values[i] = sigmoid_func(values[i]);
    }
    
}

double d_sigmoid(double x){
    // double sigm = sigmoid_func(x);

    return x*(1.0-x);
}


void softmax(size_t nb_values , double* values){

    double exp_sum = 0;

//    clip(nb_values, values, CLIP_MIN, CLIP_MAX);

    for (size_t i = 0; i < nb_values; i++)
    {   
        values[i] = exp(values[i]);
        // printf("exp %lf \n", values[i]);
        exp_sum += values[i];
    }
    
    for (size_t i = 0; i < nb_values; i++)
    {
        values[i] /= exp_sum;
    }


}

double d_softmax(double x){

    return x*(1.0-x);

}


Activation_derivative resolve_derivative(Activation_func func){

    // unforutnately I cannot do a switch statement :(

    if (func == relu){
        return d_relu;
    } else if (func == sigmoid){
        return d_sigmoid;
    }else if (func == leaky_relu){
        return d_leaky_relu;
    } else{
        fprintf(stderr, "No derivative known for function provided \n");
        return NULL;
    }

}

Output_Activation_derivative resolve_out_derivative(Output_Activation_func func){

    if(func == sigmoid_output){
        return d_sigmoid;
    }else if (func == softmax){
        return d_softmax;
    }else{
        fprintf(stderr, "No derivative known for given output activation functions \n");
        return NULL;
    }

}
