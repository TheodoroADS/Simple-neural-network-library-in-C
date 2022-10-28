#include <malloc.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "matrix.h"
#include <string.h>

Matrix* matrix_new(size_t nb_rows, size_t nb_cols){


    assert(nb_rows > 0 && nb_cols > 0);

    Matrix* mat = malloc(sizeof(Matrix));


    if (!mat){
        printf("Error in allocating matrix %s \n",strerror(errno));
        return NULL;
    }   


    mat->data = calloc(nb_rows,sizeof(double*));

    if (!mat->data){
        free(mat);
        fprintf(stderr , "Error in allocating matrix %s \n",strerror(errno));
        return NULL;
    }

    mat->nb_rows = nb_rows;


    for(size_t i = 0 ; i < nb_rows;  i ++){

        mat->data[i] = calloc(nb_cols, sizeof(double));
        if (!mat->data[i]){
            
            for (int j = 0; j < i ; j ++){
                free(mat->data[i]);
            }
            free(mat->data);
            free(mat);
            fprintf(stderr, "Error in allocating matrix %s \n", strerror(errno));
            return NULL;
        }
    }
    mat->nb_cols= nb_cols;

    return mat;

}

Matrix* matrix_zeros(size_t nb_rows, size_t nb_cols){

    Matrix* mat = matrix_new(nb_rows, nb_cols);


    if (!mat){
        return NULL;
    }

    printf("Size rows : %lld, Size cols: %lld \n", mat->nb_rows, mat->nb_cols);

    for (size_t i = 0; i < nb_rows; i++)
    {
        for (size_t j = 0; j < nb_cols; j++)
        {
            // printf(" %lld %lld \n", i,j);
            mat->data[i][j] = 0;
        }
        
    }

    return mat;
    
}

Matrix* as_vector(size_t dim, double* values){

    Matrix* vec = matrix_new(1, dim);
    
    if(!vec){
        fprintf(stderr, "Could not create vector \n");
        return NULL;
    }   

    for (size_t i = 0; i < dim; i++)
    {
        vec->data[0][i] = values[i];
    }
    
    return vec;
}

Matrix* as_batch(size_t batch_size, size_t dim, double** values, Matrix* batch){

    if(!batch){
        
        batch = matrix_new(batch_size, dim);

        if(!batch){
            fprintf(stderr, "Could not create batch \n");
            return NULL;
        }
    }


    for (size_t i = 0; i < batch_size; i++)
    {
        for (size_t j = 0; j < dim; j++)
        {
            batch->data[i][j] = values[i][j];
        }
    }

    
    return batch;
}



Matrix* matrix_random(size_t nb_rows, size_t nb_cols){
    
    Matrix* mat = matrix_new(nb_rows, nb_cols);

    if (!mat){
        return NULL;
    }

    for (size_t i = 0; i < nb_rows; i++)
    {
        for (size_t j = 0; j < nb_cols; j++)
        {
            mat->data[i][j] = (rand()/(((double) RAND_MAX + 1)/2) - 1);
        }
        
    }

    return mat;
}

void matrix_delete(Matrix* mat){

    for (size_t i = 0; i < mat->nb_rows; i++)
    {
        free(mat->data[i]);
    }

    free(mat->data);
    free(mat);

}


void matrix_apply(Matrix* mat, void (*func)(double*)){

    for (size_t i = 0; i < mat->nb_rows; i++)
    {
        for (size_t j = 0; j < mat->nb_cols; j++)
        {
            func(&mat->data[i][j]);
        }
        
    }
}



void matrix_apply_rows(Matrix* mat,void (*func)(size_t, double*)){

    for (size_t i = 0; i < mat->nb_rows; i++)
    {
        func(mat->nb_cols, mat->data[i]);
    }
    
}


void matrix_render(Matrix* mat){

    printf("Matrix: %lld X %lld :\n", mat->nb_rows, mat->nb_cols);

    for (size_t i = 0; i < mat->nb_rows; i++)
    {
        for (size_t j = 0; j < mat->nb_cols; j++)
        {
            printf(" %lf ", mat->data[i][j]);
        }

        printf("\n");
        
    }
    printf("\n");
}


void matrix_mul(Matrix* A, Matrix* B, Matrix* Res){

    // printf("Matrix mul: [%lld, %lld] * [%lld, %lld] = [%lld, %lld]\n",
    //  A->nb_rows, A->nb_cols, B->nb_rows, B->nb_cols, Res->nb_rows, Res->nb_cols);


    assert(A->nb_cols == B->nb_rows);
    assert(A->nb_rows == Res->nb_rows && B->nb_cols == Res->nb_cols);

    for (size_t i = 0; i < A->nb_rows; i++)
    {
        for (size_t j = 0; j < B->nb_cols; j++)
        {   
            Res->data[i][j] = 0;

            for (size_t k = 0; k < A->nb_cols; k++)
            {
                Res->data[i][j] += A->data[i][k]*B->data[k][j]; 
            }
        }
    }


}


void matrix_add(Matrix* M1, Matrix* M2){

    // printf("Matrix add: [%lld, %lld] + [%lld, %lld] \n",
    //  M1->nb_rows, M1->nb_cols, M2->nb_rows, M2->nb_cols);


    if(M1->nb_cols == M2->nb_cols && M1->nb_rows == M2->nb_rows){

        for (size_t i = 0; i < M1->nb_rows; i++)
        {
            for (size_t j = 0; j < M1->nb_cols; j++)
            {
                M1->data[i][j] += M2->data[i][j];
            }
            
        }   


    }else if (M2->nb_cols == 1){
        assert(M1->nb_rows == M2->nb_rows);


        for (size_t i = 0; i < M1->nb_rows; i++)
        {
            for (size_t j = 0; j < M1->nb_cols; j++)
            {
                M1->data[i][j] += M2->data[i][0];
            }
            
        }
        
    }else if(M2->nb_rows == 1 ){

        assert(M1->nb_cols == M2->nb_cols);

        for (size_t i = 0; i < M1->nb_rows; i++)
        {
            for (size_t j = 0; j < M1->nb_cols; j++)
            {
                M1->data[i][j] += M2->data[0][j];
            }
            
        }
        

    }else{
        fprintf(stderr, "Fatal: Incompatible shapes! M1(%lld,%lld) , M2(%lld,%lld) \n" , M1->nb_rows, M1->nb_cols, M2->nb_rows, M2->nb_cols);
        exit(1);
    }
    
}


size_t matrix_argmax(Matrix* mat, size_t axis, size_t pos){

    assert(axis < 2 && axis >= 0);

    double max;
    size_t argmax = 0;

    if(axis == 0){

        assert(pos >= 0 && pos < mat->nb_rows);
        
        max = mat->data[pos][0];

        for (size_t i = 1; i < mat->nb_cols; i++)
        {   
            if (mat->data[pos][i] > max){
                max = mat->data[pos][i];
                argmax = i;
            }
        }
        

    }else{

        assert(pos >= 0 && pos < mat->nb_cols);

        max = mat->data[0][pos];

        for (size_t i = 1; i < mat->nb_rows; i++)
        {   
            if (mat->data[i][pos] > max){
                max = mat->data[i][pos];
                argmax = i;
            }
        }

    }   

    return argmax;
}


Matrix* matrix_transposed(Matrix* mat){

    Matrix* transposed = matrix_new(mat->nb_rows, mat->nb_cols);

    for (size_t i = 0; i < mat->nb_rows; i++)
    {
        for (size_t j = 0; j < mat->nb_cols; j++)
        {
            transposed->data[j][i] = mat->data[i][j];
        }
    }

    return transposed;
}

/*

int main(void){

    Matrix* mat = matrix_random(2,2 );

    matrix_render(mat);

    // mat->data[0][0] = 2;

    // printf("jooj %d \n", mat->data[0][0] == 2);

    // matrix_render(mat);

    // matrix_apply(mat, times2);

    // matrix_render(mat);

    Matrix* matheus = matrix_random(2,2);

    printf("matheus \n");

    matrix_render(matheus);

    Matrix* mama = matrix_new(2,2);

    matrix_mul(mat, matheus, mama);


    matrix_render(mama);

    matrix_add(mama, matheus);

    matrix_render(mama);

    Matrix* transposed = matrix_transposed(mama);

    matrix_render(transposed);

    matrix_delete(mama);
    matrix_delete(matheus);
    matrix_delete(mat);
    matrix_delete(transposed);
    return 0;
}*/

