#pragma once

#include "loss.h"
#include "activation.h"

typedef struct Optimizer{
        
    Output_Activation_derivative out_activation_d;
    Loss_derivative loss_d;

    float (*gradient_generator)(struct Optimizer*,float,float);

} Optimizer;


float optimizer_default(Optimizer* opt , float layer_val, float reference_val);

float optimizer_cce_and_softmax(Optimizer* opt , float layer_val, float reference_val);

Optimizer NN_get_optimizer(Output_Activation_derivative d_out, Loss_derivative d_loss);
