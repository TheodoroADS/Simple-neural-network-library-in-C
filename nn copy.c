#include "matrix.h"
#include "nn.h"
#include <stdio.h>
#include <stdlib.h>
#include "loss.h"
#include <assert.h> 

#define INITIAL_LAYER_CAPACITY 3

#define LEARNING_RATE = 0.1

#define WHITE "\033[0;37m"
#define GREEN "\033[0;32m"
#define CYAN "\033[0;36m"
#define PURPLE "\033[0;35m"

void foward(Hidden_layer layer,Matrix* input, Matrix* output){

    // printf(GREEN);
    // printf("input: [%lld, %lld] weights: [%lld, %lld]\n", input->nb_rows, input->nb_cols, layer.weights.nb_rows, layer.weights.nb_cols);
    // printf(WHITE);

    matrix_mul(input,  &layer.weights, output);
    matrix_add(output, &layer.biases);
    matrix_apply(output, layer.activation);

}


NN NN_create(int input_size, int batch_size, int output_size, Output_Activation_func output_activation, Loss_func loss_function)
{

    Matrix *inputLayer , *outputLayer;
    NN instance;

    Hidden_layer** hiddenLayers;

    Loss_derivative d_loss = get_loss_derivative(loss_function);

    if(!d_loss){
        exit(1);
    }

    if (!(inputLayer = matrix_new(batch_size, input_size)) || !(outputLayer = matrix_new(batch_size,output_size))){
        fprintf(stderr, "Fatal: could not allocate input or output matrix \n");
        exit(1);
    }

    hiddenLayers = malloc(INITIAL_LAYER_CAPACITY * sizeof(Hidden_layer*));

    if(!hiddenLayers){
        fprintf(stderr,"Fatal: could not allocate %d hidden layers \n", INITIAL_LAYER_CAPACITY);
        exit(1);
    }


    instance.inputs = inputLayer;
    instance.outputs = outputLayer;
    instance.batch_size = batch_size;
    instance.output_layer_weights = NULL;
    instance.hidden_layers = hiddenLayers;
    instance.hidden_layer_count = 0;
    instance.allocated_layers = INITIAL_LAYER_CAPACITY;
    instance.output_activation = output_activation;
    instance.loss_function = loss_function;
    instance.d_loss_function = d_loss;
    instance.ready =0;


    return instance;
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

    layer_values = matrix_new(network->batch_size, layer_size);

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

    // printf("reviver porco \n");
    Matrix* output_layer_weights = matrix_random(get_last_layer(network)->values.nb_cols, network->outputs->nb_cols);

    // printf("é muito mais importante \n");

    if(!output_layer_weights){
        fprintf(stderr, "Error compiling network: could not allocate output weights matrix \n");
        return 0;
    } 

    network->output_layer_weights = output_layer_weights;
    network->ready = 1;
    // printf("Do que porco \n");
    return 1;
}

//TODO create function to free network memory

static void NN_feed_foward(NN* network, Matrix* input){

    assert(network->ready);
    assert(network->hidden_layer_count > 0);

    foward(*(network->hidden_layers[0]), input, &network->hidden_layers[0]->values);

    for (size_t i = 1; i < network->hidden_layer_count; i++)
    {
        foward(*network->hidden_layers[i], & network->hidden_layers[i -1]->values, &network->hidden_layers[i]->values);
    }

    matrix_mul(&get_last_layer(network)->values, network->output_layer_weights, network->outputs);

    matrix_apply_column(network->outputs, 0 , network->output_activation);

}

double NN_eval_loss(NN* network, double** reference_vals){

    assert(network->ready);

    double loss = 0;


    for(size_t i = 0; i < network->batch_size; i++){
        loss += network->loss_function(network->outputs->nb_cols, network->outputs->data[i], reference_vals[i]);
    }

    return loss / network->batch_size;
}   




static size_t get_gradients_array_size(NN* network){

    size_t max_layer_size;
    double *gradient_array;


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

    }


    for (size_t i = 0; i < batch_size; i++)
    {
        vectors[i] = calloc(output_size, sizeof(double));
        if (vectors[i])
        {
            for (size_t j = 0; j < output_size; j++)
            {
                vectors[i][j] = 0;
            }
            
        vectors[i][labels[i]] = 1;
        
        }else{
            fprintf(stderr, "Could not allocate one of the vectors for one hot representation \n");
            exit(1);
        }
    }

    return vectors;
    
}



static inline void flip_gradients(double** gradients1, double** gradients2){
    
    double* temp;

    temp = *gradients1;
    *gradients1 = *gradients2;
    *gradients2 = temp;
}

static void backpropagate(NN* network, double learning_rate,double** reference_vals, double* layer_gradients1, double* layer_gradients2 ){

    // printf("Cheguei aqui \n");

    size_t gradients_size = network->outputs->nb_cols;

    // matrix_render(network->outputs);

    //---------- output layer gradients ----------------------------

    // printf("output layer activation gradient \n");
    for (size_t i = 0; i < gradients_size; i++)
    {
        // calculating the average loss for the batch
        
        double avg = 0;

        // printf("gostosinho \n");

        //NOTE: for now I am doing somthing that only work for categorical cross entropy loss and softmax output layer activation
        for (size_t j = 0; j < network->outputs->nb_rows; j++)
        {
            avg += network->outputs->data[j][i] - reference_vals[j][i]; 
        }
        // printf("bunitinho \n");
        
        avg /= network->batch_size;

        // printf("%lf ", avg);

        //calculating the dloss/dy 

        layer_gradients1[i] = avg;
    }

    // printf("paratin pin pin \n");

    //updating output weights
    for (size_t i = 0; i < get_last_layer(network)->values.nb_cols ; i++)
    {       
            double correction = 0;
        
            for (size_t j = 0; j < get_last_layer(network)->values.nb_rows; j++)
            {
                correction += get_last_layer(network)->values.data[j][i];
            }
            
            // printf("kiek in de kok \n");
            correction /= network->batch_size;

            for (size_t k = 0; k < network->output_layer_weights->nb_cols; k++)
            {
                network->output_layer_weights->data[i][k] -= learning_rate*layer_gradients1[k]*correction;
                
            }

            // printf("kok in de kiek \n");

            
    }

    // printf("koko \n");

    //computing gradients for the last hidden layer's activations
    //NOTE: inside of layer_gradients1[i] is dactivation[i]/Z[i] * dloss/dactivaton[i] because of specific case of CCE + softmax
    
    for (size_t i = 0; i < get_last_layer(network)->values.nb_cols; i++)
    {
        layer_gradients2[i] = 0;
        // printf("koka \n");

        for (size_t j = 0; j < gradients_size; j++)
        {   
            //BUNDA MOLE E SECA 
            layer_gradients2[i] += network->output_layer_weights->data[i][j]*layer_gradients1[j];

        } 

    // printf("kaka \n");


    }
    
    // printf("vishhhhh \n");
    
    //flipping gradients arrays
    flip_gradients(&layer_gradients1, &layer_gradients2);

    // printf("é muita treta \n");

    // printf("outputs: \n");
    // matrix_render(network->outputs);

    // printf("weights: \n");
    // matrix_render(network->output_layer_weights);

    //backpropagating to the rest of the layers
    for (int layer_idx = network->hidden_layer_count - 1; layer_idx >= 0; layer_idx--)
    {
        // printf("viver num lugar \n");
        
        // printf("layer idx %lld \n", layer_idx);

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

        // printf("onde ninguem \n");


        for (size_t i = 0; i < gradients_size; i++)
        {
            double avg_derivative = 0;

            for (size_t j = 0; j < network->batch_size; j++)
            {
                avg_derivative += layer->d_activation(layer->values.data[j][i]);
            }

            avg_derivative /= network->batch_size;
            
            layer_gradients1[i] *= avg_derivative;
        }
        
        // printf("te respeita \n");

        //adjusting biases
        for (size_t i = 0; i < layer->biases.nb_cols; i++)
        {
            layer->biases.data[0][i] -= learning_rate *layer_gradients1[i];
        }
        
        // printf("racionais mc \n");

        //adjusting weights
        for (size_t i = 0; i < previous_layer_activations->nb_cols ; i++)
            {       
                double correction = 0;
                
                // printf("bunda mole\n");

                for (size_t j = 0; j < previous_layer_activations->nb_rows; j++)
                {
                    correction += previous_layer_activations->data[j][i];
                }
                

                correction /= network->batch_size;

                for (size_t k = 0; k < layer->values.nb_cols; k++)
                {   
                    // printf("coco k %lld i %lld \n", k, i);
                    layer->weights.data[i][k] -= learning_rate*layer_gradients1[k]*correction;
                    // printf("zinho \n");

                }


            }

        // printf("caetano veloso \n");

        if(layer_idx > 0){

            //computing gradients for the preivous layer activations
            //NOTE: inside of layer_gradients1[i] is dactivation[i]/Z[i] * dloss/dactivaton[i]
            
            for (size_t i = 0; i < previous_layer_activations->nb_cols; i++)
            {
                layer_gradients2[i] = 0;

                for (size_t j = 0; j < gradients_size; j++)
                {
                    layer_gradients2[i] += layer->weights.data[i][j]*layer_gradients1[j];
                }
            }
                
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
        //     printf("%lf ", layer_gradients1[cock]);
        // }
        
        flip_gradients(&layer_gradients1, &layer_gradients2);
        

        // printf("cock \n");
        }

        
        //this should be it!
        // exit(0);
    
}



void NN_fit_classification(NN* network,size_t nb_examples ,size_t nb_epochs ,double** values, int* labels, double learning_rate){

    size_t layer_gradients_size = get_gradients_array_size(network);
    double* layer_gradients1 =  calloc(layer_gradients_size, sizeof(double));
    double* layer_gradients2 =  calloc(layer_gradients_size, sizeof(double));
    double** one_hots = NULL;
    Matrix* batch = NULL;

    if(!layer_gradients1 || !layer_gradients2){
        fprintf(stderr, "Could not allocate memory for backpropagation \n");
        exit(1);
    }  

    for (size_t epoch = 0; epoch < nb_epochs; epoch++)
    {
        printf("Training: epoch %lld/%lld... " , epoch, nb_epochs );

        int* labels_ptr = labels;

        // printf("Vamo pro as_batch \n");

        batch = as_batch(network->batch_size, network->inputs->nb_cols, values, batch);

        // printf("Saimos do as_batch \n");

        // matrix_render(batch);

        double average_loss = 0;
        double loss;

        for (size_t batch_num = network->batch_size; batch_num < nb_examples - network->batch_size; batch_num += network->batch_size)
        {   
            // printf("vamos pro to_onehot \n");

            one_hots = to_onehot(network->batch_size, network->outputs->nb_cols, labels_ptr, one_hots);
            // printf("saimos do to_onehot \n");

            // printf("vamos pro feed foward  \n");

            
            NN_feed_foward(network, batch);

            // printf("saimos do feed foward  \n");
            
            // printf("vamos pro eval loss\n");

            loss = NN_eval_loss(network, &values[batch_num]);
            // printf("loss : %lf \n", loss);
            // matrix_render(network->outputs);
            average_loss += loss;
            
            // printf("saimos do eval loss\n");

            // printf("vamos pro backpropagate\n");
            backpropagate(network, learning_rate, one_hots, layer_gradients1, layer_gradients2);

            // printf("saimos do backpropagate\n");
            batch = as_batch(network->batch_size, network->inputs->nb_cols, &values[batch_num], batch);

        }
        
        average_loss /= (int) (nb_examples / network->batch_size);

        printf("average loss: %lf \n", average_loss);
        
    }
    
    matrix_delete(batch);
    free(layer_gradients1);
    free(layer_gradients2);
    
    for (size_t i = 0; i < network->batch_size; i++)
    {
        free(one_hots[i]);
    }
    
    free(one_hots);

}


int* NN_predict_class_batch(NN* network, Matrix* input){

    assert(input->nb_cols == network->inputs->nb_cols);
    assert(input->nb_rows == network->batch_size);

    int* predictions = calloc(network->outputs->nb_cols, sizeof(int));

    if(!predictions){
        fprintf(stderr, "Could not allocate memory for preficitions \n");
        return NULL;
    }

    NN_feed_foward(network, input);
    
    matrix_render(network->outputs);

    for (size_t i = 0; i < network->batch_size; i++)
    {
        predictions[i] = matrix_argmax(network->outputs, 0, i);
    }
    
    return predictions;
}


