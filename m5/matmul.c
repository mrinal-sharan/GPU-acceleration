#include "matmul.h"


static bool shape_ok(const Matrix *restrict a, const Matrix *restrict b, Matrix *restrict c){

    if (a->cols != b->rows ) return false;
    if (a->rows != c->rows) return false;
    if (c->cols != b->cols) return false;

    return true;

}

bool my_matmul(const Matrix *restrict a, const Matrix *restrict b, Matrix *restrict c){

    size_t R = a->rows;   // a is R × F
    size_t F = a->cols;
    size_t K = b->cols;   // b is F × K

    if (!shape_ok(a, b, c)){

        return false;
    }

    
    for (size_t i=0; i<R; i++){
        for (size_t k=0; k<F; k++){
            for (size_t j=0; j<K; j++){
                c->data[i * K + j] += a->data[i * F + k] * b->data[k * K + j];

            }

        }
    }


    return true;


    

}

bool ref_matmul(const Matrix *restrict a, const Matrix *restrict b, Matrix *restrict c){

    size_t R = a->rows;   // a is R × F
    size_t F = a->cols;
    size_t K = b->cols;   // b is F × K

    if (!shape_ok(a, b, c)){

        return false;
    }

    double sum = 0;

    for (size_t i=0; i<R; i++){
        for (size_t j=0; j<K; j++){
            for (size_t k=0; k<F; k++){
                sum += (double)a->data[i * F + k] * (double)b->data[k * K + j];
            }
            c->data[i * K + j] = (float)sum;
            sum = 0;

        }
    }

    return true;
    

    

}
