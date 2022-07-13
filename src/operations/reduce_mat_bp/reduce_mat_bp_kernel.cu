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

#include "reduce_mat_bp.h"

// clang-format off
template<ReduceMatrixOperation operation, bool across_batch>
__global__ void reduce_mat_bp_kernel(
    const float* __restrict__ A,
          float* __restrict__ A_grd,
    const float* __restrict__ B,
    const float* __restrict__ B_grd,
    unsigned int          bat_size,
    unsigned int          vec_size){
    // clang-format on

    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idy >= bat_size)
        return;

    if (idx >= vec_size)
        return;

    if constexpr (across_batch){
        if constexpr (operation == REDUCE_MATRIX_OP_SUM) {
            A_grd[INDEX_2D(vec_size, idy, idx)] += B_grd[idx];
        } else if constexpr (operation == REDUCE_MATRIX_OP_MEAN) {
            A_grd[INDEX_2D(vec_size, idy, idx)] += B_grd[idx] / bat_size;
        } else if constexpr (operation == REDUCE_MATRIX_OP_MIN) {
            A_grd[INDEX_2D(vec_size, idy, idx)] += B_grd[idx] * (B[idx] == A[INDEX_2D(vec_size, idy, idx)]);
        } else if constexpr (operation == REDUCE_MATRIX_OP_MAX) {
            A_grd[INDEX_2D(vec_size, idy, idx)] += B_grd[idx] * (B[idx] == A[INDEX_2D(vec_size, idy, idx)]);
        }
    }else{
        if constexpr (operation == REDUCE_MATRIX_OP_SUM) {
            A_grd[INDEX_2D(vec_size, idy, idx)] += B_grd[idy];
        } else if constexpr (operation == REDUCE_MATRIX_OP_MEAN) {
            A_grd[INDEX_2D(vec_size, idy, idx)] += B_grd[idy] / vec_size;
        } else if constexpr (operation == REDUCE_MATRIX_OP_MIN) {
            A_grd[INDEX_2D(vec_size, idy, idx)] += B_grd[idy] * (B[idy] == A[INDEX_2D(vec_size, idy, idx)]);
        } else if constexpr (operation == REDUCE_MATRIX_OP_MAX) {
            A_grd[INDEX_2D(vec_size, idy, idx)] += B_grd[idy] * (B[idy] == A[INDEX_2D(vec_size, idy, idx)]);
        }
    }
}

template __global__ void
    reduce_mat_bp_kernel<REDUCE_MATRIX_OP_SUM, false>(const float* __restrict__ A,
                                                      float* __restrict__ A_grd,
                                                      const float* __restrict__ B,
                                                      const float* __restrict__ B_grd,
                                                      unsigned int bat_size,
                                                      unsigned int vec_size);
template __global__ void
    reduce_mat_bp_kernel<REDUCE_MATRIX_OP_MEAN, false>(const float* __restrict__ A,
                                                       float* __restrict__ A_grd,
                                                       const float* __restrict__ B,
                                                       const float* __restrict__ B_grd,
                                                       unsigned int bat_size,
                                                       unsigned int vec_size);
template __global__ void
    reduce_mat_bp_kernel<REDUCE_MATRIX_OP_MIN, false>(const float* __restrict__ A,
                                                      float* __restrict__ A_grd,
                                                      const float* __restrict__ B,
                                                      const float* __restrict__ B_grd,
                                                      unsigned int bat_size,
                                                      unsigned int vec_size);
template __global__ void
    reduce_mat_bp_kernel<REDUCE_MATRIX_OP_MAX, false>(const float* __restrict__ A,
                                                      float* __restrict__ A_grd,
                                                      const float* __restrict__ B,
                                                      const float* __restrict__ B_grd,
                                                      unsigned int bat_size,
                                                      unsigned int vec_size);

template __global__ void
    reduce_mat_bp_kernel<REDUCE_MATRIX_OP_SUM, true>(const float* __restrict__ A,
                                                     float* __restrict__ A_grd,
                                                     const float* __restrict__ B,
                                                     const float* __restrict__ B_grd,
                                                     unsigned int bat_size,
                                                     unsigned int vec_size);
template __global__ void
    reduce_mat_bp_kernel<REDUCE_MATRIX_OP_MEAN, true>(const float* __restrict__ A,
                                                      float* __restrict__ A_grd,
                                                      const float* __restrict__ B,
                                                      const float* __restrict__ B_grd,
                                                      unsigned int bat_size,
                                                      unsigned int vec_size);
template __global__ void
    reduce_mat_bp_kernel<REDUCE_MATRIX_OP_MIN, true>(const float* __restrict__ A,
                                                     float* __restrict__ A_grd,
                                                     const float* __restrict__ B,
                                                     const float* __restrict__ B_grd,
                                                     unsigned int bat_size,
                                                     unsigned int vec_size);
template __global__ void
    reduce_mat_bp_kernel<REDUCE_MATRIX_OP_MAX, true>(const float* __restrict__ A,
                                                     float* __restrict__ A_grd,
                                                     const float* __restrict__ B,
                                                     const float* __restrict__ B_grd,
                                                     unsigned int bat_size,
                                                     unsigned int vec_size);