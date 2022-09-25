#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_CSV_READ 200000

#define SIZE 28

#define HOW_MANY 1

void read_csv_mnist(FILE* input_file){

    int row = 0;            
    int col;                                                                                                            
    char line[MAX_CSV_READ];

    fgets(line, sizeof(line) , input_file); //ignoring the first line

    while(fgets(line, sizeof(line) , input_file) && row < HOW_MANY ){

        FILE* output_file = fopen("img.ppm", "w");

        fprintf(output_file, "P3 %d %d 255 \n", SIZE, SIZE);

        char* value; //the value separated between comas

        value = strtok(line, ",");  

        int pixel;

        int label = atoi(value);


        col = 0;

        while(value != NULL){
            pixel = atoi(value);
            fprintf(output_file, "%d %d %d ", pixel, pixel, pixel);
            value = strtok(NULL, ",");
            col++;
        }   

        row++;

    }

}


int main(){


    FILE* csv_file = fopen("mnist_train.csv", "r");

    read_csv_mnist(csv_file);

    fclose(csv_file);


}