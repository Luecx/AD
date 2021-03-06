//
// Created by Luecx on 02.07.2022.
//
#include "softmax_bp.h"

#define MATRIX_INDEX(size_x, m, n) (m * size_x + n)

__global__ void softmax_bp_kernel(
          float* __restrict__ A_grd,
    const float* __restrict__ B,
    const float* __restrict__ B_grd,
    unsigned int m,
    unsigned int n){

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx >= n){
        return;
    }
    if (idy >= m){
        return;
    }

    int offset = MATRIX_INDEX(n, idy, 0);
    int j = idx;

    float gradient = 0;
    for (int i = 0; i < n; i++){
        if(i == j){
            gradient += B_grd[offset + i] * B[offset + j] * (1 - B[offset + i]);
        }else{
            gradient += B_grd[offset + i] * B[offset + j] * (0 - B[offset + i]);
        }
    }
    A_grd[offset + j] += gradient;




}
