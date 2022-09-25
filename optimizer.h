#pragma once

#include "loss.h"
#include "activation.h"

typedef struct Optimizer{
        
    Output_Activation_derivative out_activation_d;
    Loss_derivative loss_d;

    double (*gradient_generator)(struct Optimizer*,double,double);

} Optimizer;


double optimizer_default(Optimizer* opt , double layer_val, double reference_val);

double optimizer_cce_and_softmax(Optimizer* opt , double layer_val, double reference_val);

Optimizer NN_get_optimizer(Output_Activation_derivative d_out, Loss_derivative d_loss);
