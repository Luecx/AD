/**
    CudAD is a CUDA neural network trainer, specific for chess engines.
    Copyright (C) 2022 Finn Eggers

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#include "reduce_mat.h"

// clang-format off
template<ReduceMatrixOperation operation, bool across_batch>
__global__ void reduce_mat_kernel(
    const float* __restrict__ A,
          float* __restrict__ B,
    unsigned int          bat_size,
    unsigned int          vec_size){
    // clang-format on

    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if constexpr (across_batch){
        if (idy >= vec_size)
            return;
    }else{
        if (idy >= bat_size)
            return;
    }

    float temp;
    if constexpr (operation == REDUCE_MATRIX_OP_SUM ||
                  operation == REDUCE_MATRIX_OP_MEAN) {
        temp = 0;
    } else if constexpr (operation == REDUCE_MATRIX_OP_MIN ||
                         operation == REDUCE_MATRIX_OP_MAX) {
        if constexpr (across_batch){
            temp = A[INDEX_2D(vec_size, 0, idy)];
        }else{
            temp = A[INDEX_2D(vec_size, 0, idy)];
        }
    }
        
    for (int idx = 0; idx < (across_batch ? bat_size : vec_size); idx++) {

        int mat_index;
        if constexpr (across_batch){
            mat_index = INDEX_2D(vec_size, idx, idy);
        }else{
            mat_index = INDEX_2D(vec_size, idy, idx);
        }

        if constexpr (operation == REDUCE_MATRIX_OP_SUM) {
            temp += A[mat_index];
        } else if constexpr (operation == REDUCE_MATRIX_OP_MEAN) {
            temp += A[mat_index];
        } else if constexpr (operation == REDUCE_MATRIX_OP_MIN) {
            temp = min(A[mat_index], temp);
        } else if constexpr (operation == REDUCE_MATRIX_OP_MAX) {
            temp = max(A[mat_index], temp);
        }
    }

    if constexpr (operation == REDUCE_MATRIX_OP_SUM) {
        B[idy] = temp;
    } else if constexpr (operation == REDUCE_MATRIX_OP_MEAN) {
        if constexpr (across_batch){
            B[idy] = temp / bat_size;
        }else{
            B[idy] = temp / vec_size;
        }
    } else if constexpr (operation == REDUCE_MATRIX_OP_MIN) {
        B[idy] = temp;
    } else if constexpr (operation == REDUCE_MATRIX_OP_MAX) {
        B[idy] = temp;
    }
}

template void __global__ reduce_mat_kernel<REDUCE_MATRIX_OP_SUM, true>(const float* __restrict__ A,
                                                                 float* __restrict__ B,
                                                                 unsigned int bat_size,
                                                                 unsigned int vec_size);
template void __global__ reduce_mat_kernel<REDUCE_MATRIX_OP_MEAN, true>(const float* __restrict__ A,
                                                                  float* __restrict__ B,
                                                                  unsigned int bat_size,
                                                                  unsigned int vec_size);
template void __global__ reduce_mat_kernel<REDUCE_MATRIX_OP_MIN, true>(const float* __restrict__ A,
                                                                 float* __restrict__ B,
                                                                 unsigned int bat_size,
                                                                 unsigned int vec_size);
template void __global__ reduce_mat_kernel<REDUCE_MATRIX_OP_MAX, true>(const float* __restrict__ A,
                                                                 float* __restrict__ B,
                                                                 unsigned int bat_size,
                                                                 unsigned int vec_size);

template void __global__ reduce_mat_kernel<REDUCE_MATRIX_OP_SUM, false>(const float* __restrict__ A,
                                                                       float* __restrict__ B,
                                                                       unsigned int bat_size,
                                                                       unsigned int vec_size);
template void __global__ reduce_mat_kernel<REDUCE_MATRIX_OP_MEAN, false>(const float* __restrict__ A,
                                                                        float* __restrict__ B,
                                                                        unsigned int bat_size,
                                                                        unsigned int vec_size);
template void __global__ reduce_mat_kernel<REDUCE_MATRIX_OP_MIN, false>(const float* __restrict__ A,
                                                                       float* __restrict__ B,
                                                                       unsigned int bat_size,
                                                                       unsigned int vec_size);
template void __global__ reduce_mat_kernel<REDUCE_MATRIX_OP_MAX, false>(const float* __restrict__ A,
                                                                       float* __restrict__ B,
                                                                       unsigned int bat_size,
                                                                       unsigned int vec_size);