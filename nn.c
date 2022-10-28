#include "matrix.h"
#include "nn.h"
#include <stdio.h>
#include <stdlib.h>
#include "loss.h"
#include <assert.h> 
#include <omp.h>
#include "optimizer.h"

#define INITIAL_LAYER_CAPACITY 3

// #define debug

// #define grad_clip //for clipping the gradients


#ifdef grad_clip

#define GRADIENTS_MAX 10000

#define GRADIENTS_MIN -GRADIENTS_MAX

#endif

void Hidden_layer_free(Hidden_layer* layer){
    matrix_delete(&layer->values);
    matrix_delete(&layer->weights);
    free(layer);
}

static inline void foward(Hidden_layer layer,Matrix* input, Matrix* output){
    
    matrix_mul(input,  &layer.weights, output);
    matrix_add(output, &layer.biases);
    matrix_apply(output, layer.activation);

}


NN* NN_create(int input_size, int batch_size, int output_size, Output_Activation_func output_activation, Loss_func loss_function)
{

    Matrix *inputLayer , *outputLayer;
    NN* instance = malloc(sizeof(NN));

    Hidden_layer** hiddenLayers;

    Loss_derivative d_loss = get_loss_derivative(loss_function);

    Output_Activation_derivative d_out = resolve_out_derivative(output_activation);

    if(!d_loss){
        exit(1);
    }

    if(!d_out){
        exit(1);
    }

    if (!(inputLayer = matrix_new(batch_size, input_size)) || !(outputLayer = matrix_new(batch_size,output_size))){
        fprintf(stderr, "Fatal: could not allocate input or output matrix \n");
        exit(1);
    }

    hiddenLayers = calloc(INITIAL_LAYER_CAPACITY , sizeof(Hidden_layer*));

    if(!hiddenLayers){
        fprintf(stderr,"Fatal: could not allocate %d hidden layers \n", INITIAL_LAYER_CAPACITY);
        exit(1);
    }

    Optimizer opt = NN_get_optimizer(d_out, d_loss);


    instance->inputs = inputLayer;
    instance->outputs = outputLayer;
    instance->batch_size = batch_size;
    instance->output_layer_weights = NULL;
    instance->hidden_layers = hiddenLayers;
    instance->hidden_layer_count = 0;
    instance->allocated_layers = INITIAL_LAYER_CAPACITY;
    instance->output_activation = output_activation;
    instance->loss_function = loss_function;
    instance->optimizer = opt;
    instance->ready =0;


    return instance;
}

void NN_free(NN* network){

    matrix_delete(network->inputs);
    matrix_delete(network->outputs);
    matrix_delete(network->output_layer_weights);
    
    for (size_t i = 0; i < network->hidden_layer_count; i++)
    {
        Hidden_layer_free(network->hidden_layers[i]);
    }
    
    free(network->hidden_layers);

    free(network);
    
}

static int ensure_capacity(NN* network, int new_capacity){

    if(network->allocated_layers < new_capacity){
        if(!realloc(network->hidden_layers, new_capacity)){
            return 0;
        }else{
            network->allocated_layers = new_capacity;
            return 1;
        }
    }

    return 1;
}


static inline Hidden_layer* get_last_layer(NN* net){
    return net->hidden_layers[net->hidden_layer_count-1];
}

int NN_add_hidden(NN* network ,int layer_size, Activation_func layer_activation){


    Matrix *layer_values, *layer_weights, * layer_biases;

    Activation_derivative d_layer_activation = resolve_derivative(layer_activation);

    if (!d_layer_activation){
        return 0;
    }

    Hidden_layer* layer = malloc(sizeof(Hidden_layer));

    if (!layer){
        fprintf(stderr, "Error adding layer: could not allocate hidden layer \n");
        return 0;
    }

    layer_values = matrix_random(network->batch_size, layer_size);

    if(network->hidden_layer_count > 0){
        layer_weights = matrix_random(get_last_layer(network)->values.nb_cols , layer_size);
    }else{
        layer_weights = matrix_random( network->inputs->nb_cols , layer_size);
    }
    

    layer_biases = matrix_zeros(1 , layer_size);

    if (!(layer_biases && layer_weights && layer_values)){
        fprintf(stderr, "Error adding layer: could not allocate matrices for layer \n");
        return 0;
    }

    layer->values = *layer_values;
    layer->weights = *layer_weights;
    layer->biases = *layer_biases;
    layer->activation = layer_activation;
    layer->d_activation = d_layer_activation;

    if(!ensure_capacity(network, network->hidden_layer_count + 1)){
        fprintf(stderr, "Error adding layer: could not allocate memory for layer pointer in NN struct \n");
        matrix_delete(&layer->values);
        matrix_delete(&layer->weights);
        matrix_delete(&layer->biases);
        free(layer);
        return 0;
    }

    network->hidden_layers[network->hidden_layer_count] = layer;
    network->hidden_layer_count++;

    return 1;
}


int NN_compile(NN* network){

    //just in case 
    assert(network->allocated_layers >= network->hidden_layer_count);
    
    assert(network->output_layer_weights == NULL);

    Matrix* output_layer_weights = matrix_random(get_last_layer(network)->values.nb_cols, network->outputs->nb_cols);


    if(!output_layer_weights){
        fprintf(stderr, "Error compiling network: could not allocate output weights matrix \n");
        return 0;
    } 

    network->output_layer_weights = output_layer_weights;
    network->ready = 1;
    // printf("Do que porco \n");
    return 1;
}


static void NN_feed_foward(NN* network, Matrix* input){

    assert(network->ready);
    assert(network->hidden_layer_count > 0);

    foward(*(network->hidden_layers[0]), input, &network->hidden_layers[0]->values);

    for (size_t i = 1; i < network->hidden_layer_count; i++)
    {
        foward(*network->hidden_layers[i], & network->hidden_layers[i -1]->values, &network->hidden_layers[i]->values);
    }

    matrix_mul(&get_last_layer(network)->values, network->output_layer_weights, network->outputs);

    matrix_apply_rows(network->outputs, network->output_activation);

}

double NN_eval_loss(NN* network, double** reference_vals){

    assert(network->ready);

    double loss = 0;


    for(size_t i = 0; i < network->batch_size; i++){
        loss += network->loss_function(network->outputs->nb_cols, network->outputs->data[i], reference_vals[i]);
    }

    return loss / network->batch_size;
}   




static size_t get_gradients_matrix_size(NN* network){

    size_t max_layer_size;

    max_layer_size = network->hidden_layers[0]->values.nb_cols;

    for(size_t i=0; i < network->hidden_layer_count ; i++){
        if(network->hidden_layers[i]->values.nb_cols > max_layer_size ){
            max_layer_size = network->hidden_layers[i]->values.nb_cols;
        }
    }   

    if(network->outputs->nb_cols > max_layer_size){
        max_layer_size = network->outputs->nb_cols;
    }   

    return max_layer_size;
}



static double** to_onehot(size_t batch_size, size_t output_size ,int* labels, double** vectors){


    if(!vectors){
        
        vectors = calloc(batch_size, sizeof(double*));
        
        if(!vectors){
            fprintf(stderr ,"Could not allocate vectors array for one hot representation \n");
            exit(1);
        }

        #pragma omp for
        for (size_t i = 0; i < batch_size; i++)
        {
            vectors[i] = calloc(output_size, sizeof(double));

            if (!vectors[i])
            {
                fprintf(stderr, "Could not allocate one of the vectors for one hot representation \n");
                exit(1);
            }
            
        }
          

    }

    #pragma omp for
    for (size_t i = 0; i < batch_size; i++)
    {
        for (size_t j = 0; j < output_size; j++)
        {
            vectors[i][j] = 0;
        }
        
         vectors[i][labels[i]] = 1;
        
    }

    return vectors;
    
}



static inline void flip_gradients(Matrix** gradients1, Matrix** gradients2){
    
    Matrix* temp;

    temp = *gradients1;
    *gradients1 = *gradients2;
    *gradients2 = temp;
}


#ifdef grad_clip

static void clip_gradients(Matrix* gradients, size_t gradients_size, double min, double max){

    for (size_t i = 0; i < gradients->nb_rows ; i++){
    
        for (size_t j = 0; j < gradients_size ; j++)
        {
            if (gradients->data[i][j] > max){
                gradients->data[i][j] = max;
            } else if (gradients->data[i][j] < min){
                gradients->data[i][j] = min;
            }
        }
        
    }

}

#endif

static void backpropagate(NN* network, double learning_rate,double** reference_vals, Matrix* layer_gradients_current, Matrix* layer_gradients_next ){


    size_t gradients_size = network->outputs->nb_cols;

    Optimizer optimizer = network->optimizer;

    // matrix_render(network->outputs);

    //---------- output layer gradients ----------------------------

    
    //calculating dloss/dy * dy/dz 
    #pragma omp for
    for (size_t example = 0; example < network->outputs->nb_rows ; example++)
    {
        for (size_t value = 0; value < gradients_size ; value++)
        {
            layer_gradients_current->data[example][value] = 
            optimizer.gradient_generator(&optimizer,network->outputs->data[example][value], reference_vals[example][value]);
        }
    }


    //updating output weights
    #pragma omp for 
    for (size_t i = 0; i < network->output_layer_weights->nb_rows; i++)
    {
        for (size_t j = 0; j < network->output_layer_weights->nb_cols; j++)
        {
            double adjustment = 0;
            
            for (size_t k = 0; k < network->batch_size; k++)
            {
                adjustment += get_last_layer(network)->values.data[k][i]* layer_gradients_current->data[k][j];
            }

            network->output_layer_weights->data[i][j] -= learning_rate*(adjustment/network->batch_size);   

        }
        
    }
    


    //computing gradients for the last hidden layer's activations
    //NOTE: inside of layer_gradients_current[i] is dactivation[i]/Z[i] * dloss/dactivaton[i] because of specific case of CCE + softmax
    #pragma omp for
    for (size_t activation = 0; activation < get_last_layer(network)->values.nb_cols; activation++)
    {

        for (size_t example = 0; example < network->batch_size; example++)
        {   
            layer_gradients_next->data[example][activation] = 0;

            for (size_t weight = 0; weight < network->output_layer_weights->nb_cols ; weight++)
            {
                layer_gradients_next->data[example][activation] += network->output_layer_weights->data[activation][weight]*layer_gradients_current->data[example][weight];
            }
        } 
    }

    #ifdef grad_clip

    clip_gradients(layer_gradients_next, get_last_layer(network)->values.nb_cols, GRADIENTS_MIN, GRADIENTS_MAX);

    #endif

    #ifdef debug

    matrix_render(layer_gradients_current);
    
    #endif

    //flipping gradients arrays
    flip_gradients(&layer_gradients_current, &layer_gradients_next);



    //backpropagating to the rest of the layers
    for (int layer_idx = network->hidden_layer_count - 1; layer_idx >= 0; layer_idx--)
    {

        Hidden_layer* layer = network->hidden_layers[layer_idx];
        Matrix* previous_layer_activations;

        if(layer_idx > 0){
            previous_layer_activations = &network->hidden_layers[layer_idx - 1]->values;
        }else{
            previous_layer_activations = network->inputs;
        }

        // multipliying the gradients of the next layer's by the derivative of the activation
        // because it will be used to calculate the gradients for the biases, the weights and the previous layer's activations

        gradients_size = layer->values.nb_cols;

        #pragma omp for
        for (size_t i = 0; i < gradients_size; i++)
        {

            for (size_t j = 0; j < network->batch_size; j++)
            {
                layer_gradients_current->data[j][i] *= layer->d_activation(layer->values.data[j][i]);
            }

        }
        
        #ifdef debug

        matrix_render(layer_gradients_current);

        #endif

        //adjusting biases
        #pragma omp for
        for (size_t i = 0; i < layer->biases.nb_cols; i++)
        {
            double bias_adjustment = 0;

            for (size_t j = 0; j < network->batch_size; j++)
            {
                bias_adjustment += layer_gradients_current->data[j][i];
            }
            

            layer->biases.data[0][i] -= learning_rate * (bias_adjustment/network->batch_size);
        }
        

        //adjusting weights
        #pragma omp for
        for (size_t i = 0; i < layer->weights.nb_rows; i++)
        {
            for (size_t j = 0; j < layer->weights.nb_cols; j++)
            {
                double adjustment = 0;
                
                for (size_t k = 0; k < network->batch_size; k++)
                {
                    adjustment += previous_layer_activations->data[k][i]* layer_gradients_current->data[k][j];
                }

                layer->weights.data[i][j] -= learning_rate*(adjustment/network->batch_size);   

            }
            
        }
    

        if(layer_idx > 0){

            //computing gradients for the preivous layer activations
            //NOTE: inside of layer_gradients_current[i] is dactivation[i]/Z[i] * dloss/dactivaton[i]

            for (size_t activation = 0; activation < previous_layer_activations->nb_cols; activation++)
            {

                for (size_t example = 0; example < network->batch_size; example++)
                {   
                    layer_gradients_next->data[example][activation] = 0;

                    for (size_t weight = 0; weight < layer->weights.nb_cols ; weight++)
                    {
                        layer_gradients_next->data[example][activation] += layer->weights.data[activation][weight]*layer_gradients_current->data[example][weight];
                    }
                } 
            }

            #ifdef grad_clip

            clip_gradients(layer_gradients_next, previous_layer_activations->nb_cols, GRADIENTS_MIN, GRADIENTS_MAX);

            #endif

        }

        // printf("activations: \n");
        // matrix_render(&layer->values);

        // printf("weights: \n");
        // matrix_render(&layer->weights);

        // printf("biases \n");
        // matrix_render(&layer->biases);


        // printf("\nlayer gradients \n");

        // for (size_t cock = 0; cock < gradients_size; cock++)
        // {
        //     printf("%lf ", layer_gradients_current[cock]);
        // }
        

        flip_gradients(&layer_gradients_current, &layer_gradients_next);
        

        }

}



void NN_fit_classification(NN* network,size_t nb_examples ,size_t nb_epochs ,double** values, int* labels, double learning_rate){

    size_t layer_gradients_size = get_gradients_matrix_size(network);
    Matrix* layer_gradients_current =  matrix_new(network->batch_size, layer_gradients_size);
    Matrix* layer_gradients_next =  matrix_new(network->batch_size, layer_gradients_size);
    double** one_hots = NULL;
    Matrix* batch = NULL;

    if(!layer_gradients_current || !layer_gradients_next){
        fprintf(stderr, "Could not allocate memory for backpropagation \n");
        exit(1);
    }  

    for (size_t epoch = 0; epoch < nb_epochs; epoch++)
    {
        printf("Training: epoch %lld/%lld... " , epoch + 1, nb_epochs );

        int* labels_ptr;

        double average_loss = 0;
        double loss;

        for (size_t batch_num = 0; batch_num < nb_examples - network->batch_size; batch_num += network->batch_size)
        {   
            labels_ptr = &labels[batch_num];
            one_hots = to_onehot(network->batch_size, network->outputs->nb_cols, labels_ptr, one_hots);
            batch = as_batch(network->batch_size, network->inputs->nb_cols, &values[batch_num], batch);

            // if(batch_num == 0){
                // matrix_render(batch);
            // }

            // printf("one hots \n ");

            // for (size_t b = 0; b < network->batch_size; b++)
            // {
            //     printf("\n [");

            //     for (size_t i = 0; i < network->outputs->nb_cols; i++)
            //     {
            //         printf("%lf ", one_hots[b][i]);
            //     }

            //     printf("]");
                
            // }
            
            NN_feed_foward(network, batch);

            #ifdef debug

            matrix_render(network->outputs);

            printf("output layer weights: \n");

            matrix_render(network->output_layer_weights);

            printf("last hidden layer activations: \n");

            matrix_render(&get_last_layer(network)->values);

            printf("last hidden layer biases: \n");

            matrix_render(&get_last_layer(network)->biases);

            printf("last hidden layer weights: \n");

            matrix_render(&get_last_layer(network)->weights);

            #endif


            loss = NN_eval_loss(network, &values[batch_num]);

            average_loss += loss;
            
            backpropagate(network, learning_rate, one_hots, layer_gradients_current, layer_gradients_next);
        }
        
        average_loss /= (int) (nb_examples / network->batch_size);

        printf("average loss: %lf \n", average_loss);
        
    }
    
    matrix_delete(batch);
    matrix_delete(layer_gradients_current);
    matrix_delete(layer_gradients_next);
    
    for (size_t i = 0; i < network->batch_size; i++)
    {
        free(one_hots[i]);
    }
    
    free(one_hots);

}

int* NN_predict_class_batch(NN* network, Matrix* input, int* predictions){

    assert(input->nb_cols == network->inputs->nb_cols);
    assert(input->nb_rows == network->batch_size);

    if(!predictions){

        int* predictions = calloc(network->outputs->nb_cols, sizeof(int));
        if(!predictions){
            fprintf(stderr, "Could not allocate memory for preficitions \n");
            return NULL;
        }
    }


    NN_feed_foward(network, input);
    
    // matrix_render(network->outputs);

    for (size_t i = 0; i < network->batch_size; i++)
    {
        predictions[i] = matrix_argmax(network->outputs, 0, i);
    }
    
    return predictions;
}

int NN_predict_class(NN* network, double* X){

    Matrix* batch = matrix_random(network->batch_size, network->inputs->nb_cols);

    for (size_t i = 0; i < network->inputs->nb_cols; i++){
        batch->data[0][i] = X[i];
    }

    NN_feed_foward(network, batch);

    matrix_delete(batch);

    return matrix_argmax(network->outputs, 0, 0);
}

int* NN_predict_class_all(NN* network, size_t how_many, double** values){

    Matrix* batch = NULL;
    size_t batch_num;

    int* predictions = calloc(how_many, sizeof(int));

    if(!predictions){
        fprintf(stderr, "Could not allocate memory for preficitions \n");
        return NULL;
    }

    for (batch_num = 0; batch_num < how_many - network->batch_size ; batch_num += network->batch_size)
    {
        batch = as_batch(network->batch_size, network->inputs->nb_cols, &values[batch_num], batch);
        NN_predict_class_batch(network, batch, &predictions[batch_num]);
    }

    // size_t remaining = how_many - batch_num;

    // assert(remaining == network->batch_size);

    // for (size_t i = 0; i < remaining ; i++)
    // {
    //     for (size_t j = 0; j < network->inputs->nb_cols; j++)
    //     {
    //         batch->data[i][j] = values[batch_num + i][j];
    //     }
        
    // }
    
    printf("Batch num : %lld, nb of examples: %lld \n", batch_num, how_many );
    
    matrix_delete(batch);

    return predictions;
}
