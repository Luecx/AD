//
// Created by Luecx on 02.07.2022.
//
#include "softmax.h"

#define MATRIX_INDEX(size_x, m, n) (m * size_x + n)
__global__ void softmax_kernel(
    const float* __restrict__ A,
          float* __restrict__ B,
    unsigned int m,
    unsigned int n){

    int batch_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (batch_id >= m){
        return;
    }

    float division = 0;

    int offset = MATRIX_INDEX(n, batch_id, 0);

    // input regularization
    float mmax = A[offset];
    for (int i = 1; i < n; i++){
        mmax = max(mmax, A[offset + i]);
    }

    // denominator (adjust using input regularization)
    for (int i = 0; i < n; i++){
        division += exp(A[offset + i] - mmax);
    }

    // numerator (adjust using input regularization)
    for (int i = 0; i < n; i++){
        B[offset + i] = exp(A[offset + i] - mmax) / division;
    }

}
#undef MATRIX_INDEX
