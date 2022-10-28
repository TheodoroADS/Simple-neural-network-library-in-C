#include <math.h>
#include "activation.h"
#include <stdio.h>
#include "loss.h"


#define LEAKY_RELU_COEF 0.1f

void relu(float* x){
    if(*x < 0){
        *x = 0;
    }
}

void leaky_relu(float* x){
    if(*x < 0){
        *x *= LEAKY_RELU_COEF;
    }
}

float d_relu(float x){
   if (x < 0){
       return 0.0;
   }else{
       return 1.0;
   }
}

float d_leaky_relu(float x){
    if(x <= 0){
        return LEAKY_RELU_COEF;
    }else{
        return 1.0;
    }
}

static inline float sigmoid_func(float x){
    return 1/(1+ exp(-x));
}   

void sigmoid(float* x){
    *x = sigmoid_func(*x);
}


void sigmoid_output(size_t nb_values, float* values){
    for (size_t i = 0; i < nb_values; i++)
    {
        values[i] = sigmoid_func(values[i]);
    }
    
}

float d_sigmoid(float x){
    // float sigm = sigmoid_func(x);

    return x*(1.0-x);
}


void softmax(size_t nb_values , float* values){

    float exp_sum = 0;

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

float d_softmax(float x){

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
