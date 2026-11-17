#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "matmul.h"

// Private to this file: only main reads the command line.
static bool parse_positive(const char *text, size_t *out) {
    char *end;
    errno = 0;
    long value = strtol(text, &end, 10);
    if (end == text) return false;       // no digits at all
    if (*end != '\0') return false;      // junk after the number
    if (errno == ERANGE) return false;   // too big for a long
    if (value <= 0) return false;
    *out = (size_t)value;
    return true;
}

int main(int argc, char *argv[]) {
    if (argc != 4) {
        fprintf(stderr, "usage: %s <R> <F> <K>\n", argv[0]);
        return EXIT_FAILURE;
    }
    size_t R, F, K;
    if (!parse_positive(argv[1], &R) || !parse_positive(argv[2], &F) || !parse_positive(argv[3], &K)) {
        fprintf(stderr, "error: R, F and K must be positive whole numbers\n");
        return EXIT_FAILURE;
    }

    // Known-answer test: [2x3] x [3x2], worked out by hand.
    float ta[] = {1, 2, 3, 4, 5, 6};
    float tb[] = {7, 8, 9, 10, 11, 12};
    const float expected[] = {58, 64, 139, 154};
    float tc[4] = {0};
    Matrix TA = {2, 3, ta}, TB = {3, 2, tb}, TC = {2, 2, tc};
    if (!my_matmul(&TA, &TB, &TC)) {
        fprintf(stderr, "error: known-answer test: shapes refused\n");
        return EXIT_FAILURE;
    }
    for (size_t i = 0; i < 4; i++) {
        if (tc[i] != expected[i]) {
            fprintf(stderr, "error: known-answer test failed at index %zu: got %f, expected %f\n", i, tc[i], expected[i]);
            return EXIT_FAILURE;
        }
    }

    // Shape test: [2x3] x [2x3] does not fit, so it must be refused.
    Matrix bad = {2, 3, tb};
    if (my_matmul(&TA, &bad, &TC)) {
        fprintf(stderr, "error: mismatched shapes were not refused\n");
        return EXIT_FAILURE;
    }

    // Each buffer is rows * cols * 4 bytes; refuse sizes whose product would overflow size_t.
    if (R > SIZE_MAX / sizeof(float) || F > SIZE_MAX / sizeof(float) || K > SIZE_MAX / sizeof(float) ||
        R > SIZE_MAX / (sizeof(float) * F) || F > SIZE_MAX / (sizeof(float) * K) || K > SIZE_MAX / (sizeof(float) * R)) {
        fprintf(stderr, "error: sizes too large to allocate\n");
        return EXIT_FAILURE;
    }

    float *a = malloc(R * F * sizeof(float));
    float *b = malloc(F * K * sizeof(float));
    float *c = malloc(R * K * sizeof(float));
    float *ref = malloc(R * K * sizeof(float));
    int status = EXIT_FAILURE;
    if (!a || !b || !c || !ref) {
        fprintf(stderr, "error: out of memory\n");
        goto cleanup;
    }

    memset(c, 0, R * K * sizeof(float));   // my_matmul adds into c, so it must start at zero
    for (size_t i = 0; i < R; i++)
        for (size_t k = 0; k < F; k++) a[i * F + k] = (float)(i + k + 1);
    for (size_t k = 0; k < F; k++)
        for (size_t j = 0; j < K; j++) b[k * K + j] = (float)(k + j + 7);

    Matrix A = {R, F, a}, B = {F, K, b}, C = {R, K, c}, Ref = {R, K, ref};

    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    bool ok = my_matmul(&A, &B, &C);
    clock_gettime(CLOCK_MONOTONIC, &end);
    if (!ok || !ref_matmul(&A, &B, &Ref)) {
        fprintf(stderr, "error: shapes refused\n");
        goto cleanup;
    }
    double ms = (end.tv_sec - start.tv_sec) * 1e3 + (end.tv_nsec - start.tv_nsec) / 1e6;
    printf("time taken: %.3f ms\n", ms);

    for (size_t i = 0; i < R * K; i++) {
        if (!(fabsf(c[i] - ref[i]) <= 1e-6f + 2e-6f * fabsf(ref[i]))) {
            fprintf(stderr, "mismatch at row %zu col %zu: got %f, expected %f\n", i / K, i % K, c[i], ref[i]);
            goto cleanup;
        }
    }
    printf("check: all %zu cells match\n", R * K);
    status = EXIT_SUCCESS;

cleanup:
    free(a); free(b); free(c); free(ref);   // free(NULL) is a safe no-op
    return status;
}
