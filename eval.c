// #include <stdio.h>
#include <stddef.h>
#include "eval.h"


float NN_error_rate(size_t nb_examples , int* predictions, int* labels){

    size_t incorrect = 0;

    for (size_t i = 0; i < nb_examples; i++)
    {
        if (predictions[i] != labels[i])
        {
            incorrect++;
        }
        
    }
    
    return ((float) incorrect)/nb_examples; 

}

float NN_accuracy_score(size_t nb_examples , int* predictions, int* labels){

    return 1.0 - NN_error_rate(nb_examples, predictions, labels);

}
