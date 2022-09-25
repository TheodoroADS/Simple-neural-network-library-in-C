#include "optimizer.h"


double optimizer_default(Optimizer* opt , double layer_val, double reference_val){
    return opt->loss_d(layer_val, reference_val) * opt->out_activation_d(layer_val);
}


double optimizer_cce_and_softmax(Optimizer* opt , double layer_val, double reference_val){
    return  layer_val - reference_val;
}

Optimizer NN_get_optimizer(Output_Activation_derivative d_out, Loss_derivative d_loss){

    Optimizer optimizer;

    if (d_loss == d_categorical_cross_entropy && d_out == d_softmax){
        optimizer.gradient_generator = optimizer_cce_and_softmax;
    }else{

        optimizer.loss_d = d_loss;
        optimizer.out_activation_d = d_out;
        optimizer.gradient_generator = optimizer_default;
    }

    return optimizer;
}
