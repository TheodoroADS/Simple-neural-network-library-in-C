#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "nn.h"
#include "loss.h"
#include "activation.h"
#include <malloc.h>


#define MAX_CSV_READ 200000
#define MNIST_INPUT_SIZE 784 //there are 784 pixels but 785 columns in csv: first column is label
#define MNIST_OUTPUT_SIZE 10
#define MNIST_TRAIN_SIZE 10000
#define MNIST_TEST_SIZE 1000

#define LEARNING_RATE 0.1

#define BATCH_SIZE 32
#define EPOCHS 100

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



int main(){

    FILE* csv_file = fopen("mnist_train.csv", "r");

    
    if(!csv_file){
        fprintf(stderr, "Could not open dataset file \n");
        exit(1);
    }

    // print_csv(csv_file);

    // // fclose(csv_file);

    int* labels = calloc(MNIST_TRAIN_SIZE, sizeof(int));

    if(!labels){
        fprintf(stderr , "Failed to allocate %lld bytes of memory to store dataset labels \n", sizeof(int)*MNIST_TRAIN_SIZE);
        exit(1);
    }

    double** values = calloc(MNIST_TRAIN_SIZE, sizeof(double*));

    if(!values){
        fprintf(stderr , "Failed to allocate %lld bytes of memory for values \n", sizeof(int)*MNIST_TRAIN_SIZE);
        exit(1);
    }

    for(int i = 0; i < MNIST_TRAIN_SIZE; i++){
        values[i] = calloc(MNIST_INPUT_SIZE, sizeof(double));
        if(!values[i]){
            fprintf(stderr , "Failed to allocate %lld bytes of memory for images\n", sizeof(int)*MNIST_INPUT_SIZE);
            exit(1);
        }
    }
    

    printf("Reading csv... ");

    read_csv_mnist(csv_file, labels, values);

    fclose(csv_file);

    printf("Done\n");

    // for (size_t i = 0; i < 10; i++)
    // {
    //     printf("Label: %d Data: \n", labels[i]);
    //     for (size_t j = 0; j < 10; j++)
    //     {
    //         printf("%d ", values[i][j]);
    //     }
        
    // }

    

    NN network;

    printf("Creating NN \n");

    network = NN_create(MNIST_INPUT_SIZE,
     BATCH_SIZE,
     MNIST_OUTPUT_SIZE,
     softmax,
     categorical_cross_entropy
    );
    printf("Created \n");

    printf("Adding first layer \n");

    NN_add_hidden(&network, 126, relu);
    printf("Adding second layer \n");
    NN_add_hidden(&network, 50, relu);
    // printf("Adding third layer \n");
    // NN_add_hidden(&network, 30, sigmoid);

    printf("Done \nCompiling \n");

    NN_compile(&network);

    printf("Done! \n");


    
    // double** one_hots = to_onehot(32, labels); 

    // printf("loss: %lf \n", NN_eval_loss(&network, one_hots));

    printf("Let's gooooo \n");

    NN_fit_classification(&network, MNIST_TRAIN_SIZE, EPOCHS, values, labels, LEARNING_RATE);


    Matrix* input = as_batch(BATCH_SIZE, MNIST_INPUT_SIZE, values, NULL);

    int* predictions = NN_predict_class_batch(&network, input);

    for (size_t i = 0; i < 10; i++)
    {
        printf("Prediction %lld: %d real %d\n",i,predictions[i], labels[i]);
    }    



    // for (size_t i = 0; i < network.hidden_layer_count; i++)
    // {      
    //     printf("Layer %lld biases \n", i);
    //     matrix_render(&network.hidden_layers[i]->biases);
    //     printf("Layer %lld weights \n", i);
    //     matrix_render(&network.hidden_layers[i]->weights);
    // }
    
    // printf("Output layer weights \n");
    // matrix_render(network.output_layer_weights);

    //-------------- cleanup -------------------------------

    // matrix_delete(input);
    // free(predictions);
    free(labels);

    for(int i = 0; i < MNIST_TRAIN_SIZE; i++){
        free(values[i]);
    }

    free(values);

    return 0;
}

