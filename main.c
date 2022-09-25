#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "nn.h"
#include "loss.h"
#include "activation.h"
#include <malloc.h>
#include "eval.h"


#define MAX_CSV_READ 200000
#define MNIST_INPUT_SIZE 784 //there are 784 pixels but 785 columns in csv: first column is label
#define MNIST_OUTPUT_SIZE 10
#define MNIST_TRAIN_SIZE 60000
#define MNIST_TEST_SIZE 10000

#define LEARNING_RATE 0.1

#define BATCH_SIZE 32
#define EPOCHS 30

void print_csv(FILE* input_file){

    char row[MAX_CSV_READ];

    while(fgets(row, sizeof(row) , input_file)){


        char* value; //the value separated between comas

        value = strtok(row, ",");

        while(value != NULL){
            printf("%s", value);
            value = strtok(NULL, ",");
        }   

        printf("\n");
    }

}


void read_csv_mnist(FILE* input_file, int* labels, double** values){


    int row = 0;            
    int col;                                                                                                            
    char line[MAX_CSV_READ];

    fgets(line, sizeof(line) , input_file); //ignoring the first line

    while(fgets(line, sizeof(line) , input_file) && row < MNIST_TRAIN_SIZE){

        char* value; //the value separated between comas

        value = strtok(line, ",");  

        labels[row] = atoi(value);

        col = 0;

        while(value != NULL){
            values[row][col] = ((double) atoi(value))/255;
            value = strtok(NULL, ",");
            col++;
        }   

        row++;
        

    }

}



int main(void){

    FILE* csv_file = fopen("mnist_train.csv", "r");

    FILE* csv_file_test = fopen("mnist_test.csv", "r");
    
    if(!csv_file){
        fprintf(stderr, "Could not open train set file \n");
        exit(1);
    }

    
    if(!csv_file_test){
        fprintf(stderr, "Could not open test set file \n");
        exit(1);
    }


    int* labels = calloc(MNIST_TRAIN_SIZE, sizeof(int));

    int* test_labels =  calloc(MNIST_TEST_SIZE, sizeof(int));;

    if(!labels){
        fprintf(stderr , "Failed to allocate %lld bytes of memory to store training set labels \n", sizeof(int)*MNIST_TRAIN_SIZE);
        exit(1);
    }


    
    if(!test_labels){
        fprintf(stderr , "Failed to allocate %lld bytes of memory to store test set labels \n", sizeof(int)*MNIST_TRAIN_SIZE);
        exit(1);
    }


    double** values = calloc(MNIST_TRAIN_SIZE, sizeof(double*));

    double** test_values = calloc(MNIST_TEST_SIZE, sizeof(double*));

    if(!values){
        fprintf(stderr , "Failed to allocate %lld bytes of memory for training values \n", sizeof(int)*MNIST_TRAIN_SIZE);
        exit(1);
    }

    if(!test_values){
        fprintf(stderr , "Failed to allocate %lld bytes of memory for testing values \n", sizeof(int)*MNIST_TEST_SIZE);
        exit(1);
    }

    for(int i = 0; i < MNIST_TRAIN_SIZE; i++){
        values[i] = calloc(MNIST_INPUT_SIZE, sizeof(double));
        if(!values[i]){
            fprintf(stderr , "Failed to allocate %lld bytes of memory for images\n", sizeof(int)*MNIST_INPUT_SIZE);
            exit(1);
        }
    }

    for(int i = 0; i < MNIST_TEST_SIZE; i++){
        test_values[i] = calloc(MNIST_INPUT_SIZE, sizeof(double));
        if(!test_values[i]){
            fprintf(stderr , "Failed to allocate %lld bytes of memory for images\n", sizeof(int)*MNIST_INPUT_SIZE);
            exit(1);
        }
    }

    
    read_csv_mnist(csv_file, labels, values);

    read_csv_mnist(csv_file_test, test_labels, test_values);

    fclose(csv_file);

    fclose(csv_file_test);

    NN* network;

    network = NN_create(MNIST_INPUT_SIZE,
     BATCH_SIZE,
     MNIST_OUTPUT_SIZE,
     softmax,
     categorical_cross_entropy
    );


    // NN_add_hidden(network, 126, relu);
    NN_add_hidden(network, 126, sigmoid);
    NN_add_hidden(network, 126, sigmoid);
    // NN_add_hidden(network, 256, relu);
    // NN_add_hidden(network, 126, relu);
    // printf("Adding third layer \n");
    // NN_add_hidden(network, 32, relu);


    NN_compile(network);


    NN_fit_classification(network, MNIST_TRAIN_SIZE, EPOCHS, values, labels, LEARNING_RATE);


    matrix_render(network->outputs);

    int* predictions = NN_predict_class_all(network, MNIST_TRAIN_SIZE - network->batch_size, values);

    int* predictions_test = NN_predict_class_all(network, MNIST_TEST_SIZE - network->batch_size, test_values);

    double accuracy = NN_accuracy_score(MNIST_TRAIN_SIZE - network->batch_size, predictions, labels);

    double real_accuracy = NN_accuracy_score(MNIST_TEST_SIZE - network->batch_size, predictions_test, test_labels);

    printf("Train accuracy: %lf \n", accuracy);
    
    printf("Test accuracy: %lf \n", real_accuracy);


    printf("joojinho: prediction %d real %d", NN_predict_class(network, values[0]) ,labels[0]); 

    // for (size_t i = 0; i < 30; i++)
    // {
    //     printf("prediction: %d , real : %d \n", predictions[i], labels[i]);
    // }
    

    // for (size_t i = 0; i < network.hidden_layer_count; i++)
    // {      
    //     printf("Layer %lld biases \n", i);
    //     matrix_render(&network.hidden_layers[i]->biases);
    //     printf("Layer %lld weights \n", i);
    //     matrix_render(&network.hidden_layers[i]->weights);
    // }
    
    // printf("Output layer weights \n");
    // matrix_render(network.output_layer_weights);

    // -------------- cleanup -------------------------------

    // matrix_delete(input);
    free(predictions);

    free(predictions_test);
   
    free(labels);

    free(test_labels);
    
    for(int i = 0; i < MNIST_TRAIN_SIZE; i++){
        free(values[i]);
    }
    
    for(int i = 0; i < MNIST_TEST_SIZE; i++){
        free(test_values[i]);
    }
    

    free(values);
    
    free(test_values);

    NN_free(network);

    return 0;
}

