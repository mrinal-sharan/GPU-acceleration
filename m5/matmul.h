#include <stddef.h>
#include <stdbool.h> 

#ifndef MATMUL_H      // "if MATMUL_H is NOT defined yet..."
#define MATMUL_H      // "...define it now (an empty marker)"

typedef struct {
    size_t rows;
    size_t cols;
    float* data;

} Matrix;

bool my_matmul(const Matrix *restrict a, const Matrix *restrict b, Matrix *restrict c);
bool ref_matmul(const Matrix *restrict a, const Matrix *restrict b, Matrix *restrict c);
 

 

#endif                // end of the if