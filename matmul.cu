#include <string.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <stdbool.h> 
#include <stdint.h> 
#include <time.h>
#include <math.h>

bool parse_positive(const char *text, size_t *out){
        char *end;
        errno = 0;
        long value = strtol(text, &end, 10);

        if (end == text) return false; 
        if (*end != '\0') return false;

        if (errno == ERANGE) return false;
        if (value <= 0) return false;

        *out = (size_t)value;
        //fprintf(stderr, "Parsed value: %ld\n", value);
        return true;
    }

    // a, b and c must not overlap


__global__ void my_matmul(const float *restrict a, const float *restrict b, float *restrict c, const size_t R, const size_t F, const size_t K){

    for (size_t k=0; k<F; k++){
        for (size_t j=0; j<K; j++){
            for (size_t i=0; i<R; i++){
                c[i * K + j] += a[i * F + k] * b[k * K + j];

            }

        }
    }

}

void ref_matmul(const float *restrict a, const float *restrict b, float *restrict c, const size_t R, const size_t F, const size_t K){

    double sum = 0;

    for (size_t i=0; i<R; i++){
        for (size_t j=0; j<K; j++){
            for (size_t k=0; k<F; k++){
                sum += (double)a[i * F + k] * (double)b[k * K + j];
            }
            c[i * K + j] = (float)sum;
            sum = 0;

        }
    }

}



int main(int argc, char *argv[]) {

    if (argc !=6){
        fprintf(stderr, "Usage: %s <N> <blocksize>\n", argv[0]);

        return EXIT_FAILURE;
    }

    size_t R, F, K, n, blocksize;

    struct timespec start, end;

    if (!parse_positive(argv[1], &R)){
        fprintf(stderr, "error: columns must be a positive whole number, got '%s'\n", argv[1]);
        return EXIT_FAILURE;

    }

    if (!parse_positive(argv[2], &F)){
        fprintf(stderr, "error: columns must be a positive whole number, got '%s'\n", argv[1]);
        return EXIT_FAILURE;

    }

    if (!parse_positive(argv[3], &K)){
        fprintf(stderr, "error: columns must be a positive whole number, got '%s'\n", argv[1]);
        return EXIT_FAILURE;

    }

    if (!parse_positive(argv[4], &n)) {
        fprintf(stderr, "error: N must be a positive whole number, got '%s'\n", argv[1]);
        return EXIT_FAILURE;
    }

    if (!parse_positive(argv[5], &blocksize)) {
        fprintf(stderr, "error: blocksize must be a positive whole number, got '%s'\n", argv[2]);
        return EXIT_FAILURE;
    }

    size_t blocks;
    blocks = (n+ blocksize -1)/blocksize;

    float *ptrA = NULL;
    float *ptrB = NULL;
    float *ptrC = NULL;
    float *ptrRef = NULL;

    if (F > SIZE_MAX/sizeof(float)){
        fprintf(stderr, "error: Too large to allocate \n");
        return EXIT_FAILURE;
    }

    if (R > SIZE_MAX/sizeof(float)){
        fprintf(stderr, "error: Too large to allocate \n");
        return EXIT_FAILURE;
    }

    if (K > SIZE_MAX/sizeof(float)){
        fprintf(stderr, "error: Too large to allocate \n");
        return EXIT_FAILURE;
    }

    if (R > SIZE_MAX/((sizeof(float) *F))){
        fprintf(stderr, "error: R is too large to allocate \n");
        return EXIT_FAILURE;
    }

    if (F > SIZE_MAX/(sizeof(float) *K)){
        fprintf(stderr, "error: F is too large to allocate \n");
        return EXIT_FAILURE;
    }

    if (K > SIZE_MAX/(sizeof(float) *R)){
        fprintf(stderr, "error: K is too large to allocate \n");
        return EXIT_FAILURE;
    }

    ptrA = (float *)malloc(R*F * sizeof(float));
    ptrB =(float *)malloc(F*K * sizeof(float));
    ptrC = (float *)malloc(R*K * sizeof(float));
    ptrRef = (float *)malloc(R*K * sizeof(float));

    if ((ptrA == NULL) || (ptrB == NULL) || (ptrC == NULL) || (ptrRef ==NULL)){

        fprintf(stderr, "Out of memory error\n");
        free(ptrA);
        free(ptrB);
        free(ptrC);
        return EXIT_FAILURE;
        

    } 

    float *d_a = NULL;
    check_error(cudaMalloc(&d_a, n * sizeof(float)),"alloc d_a");

    float *d_b = NULL;
    check_error(cudaMalloc(&d_b, n * sizeof(float)),"alloc d_b");

    float *d_c = NULL;
    check_error(cudaMalloc(&d_c, n * sizeof(float)),"alloc d_c");    
    memset(ptrC, 0 , R * K * sizeof(float));

    for (size_t i=0; i < R; i ++){
        for (size_t j =0; j < F; j++){

        ptrA[i * F + j] = i + j + 1;

    }
}

    for (size_t i=0; i < F; i ++){
        for (size_t j =0; j < K; j++){

        ptrB[i * K + j] = i  + j + 7;

    }
}



check_error(cudaMemcpy(d_a, ptra, n * sizeof(float), cudaMemcpyHostToDevice), "copy d_a");
check_error(cudaMemcpy(d_b, ptrb, n * sizeof(float), cudaMemcpyHostToDevice), "copy d_b");


check_error(cudaDeviceSynchronize(), "wait before kernel");
clock_gettime(CLOCK_MONOTONIC, &start);

my_matmul<<blocks,blocksize >>(ptrA, ptrB, ptrC, R, F, K);

check_error(cudaGetLastError(), "kernel_launch_error\n");
check_error(cudaDeviceSynchronize(), "wait after kernel");
clock_gettime(CLOCK_MONOTONIC, &end);

double ms = (end.tv_sec - start.tv_sec) * 1e3
          + (end.tv_nsec - start.tv_nsec) / 1e6;



check_error(cudaDeviceSynchronize(), "wait before loading c back");
check_error(cudaMemcpy(ptrc, d_c, n * sizeof(float), cudaMemcpyDeviceToHost), "copy d_c");
check_error(cudaDeviceSynchronize(), "wait after loading c back");
clock_gettime(CLOCK_MONOTONIC, &end);


printf("No. of same entries:%ld", count);

cudaFree(d_a);
cudaFree(d_b);
cudaFree(d_c);

free(ptrA);
free(ptrB);
free(ptrC);



return 0;



}