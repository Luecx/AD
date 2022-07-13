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

#include <iostream>

// clang-format off
template<ReduceMatrixOperation operation, bool across_batch>
void reduce_mat_bp_host(
    const float* A,
          float* A_grd,
    const float* B,
    const float* B_grd,
    unsigned int bat_size,
    unsigned int vec_size){
    // clang-format on

    for (int idy = 0; idy < bat_size; idy++) {
        for (int idx = 0; idx < vec_size; idx++) {

            if constexpr (across_batch){
                if constexpr (operation == REDUCE_MATRIX_OP_SUM) {
                    A_grd[INDEX_2D(vec_size, idy, idx)] += B_grd[idx];
                } else if constexpr (operation == REDUCE_MATRIX_OP_MEAN) {
                    A_grd[INDEX_2D(vec_size, idy, idx)] += B_grd[idx] / vec_size;
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
    }
}

template void reduce_mat_bp_host<REDUCE_MATRIX_OP_SUM, false>(const float* A,
                                                       float*       A_grd,
                                                       const float* B,
                                                       const float* B_grd,
                                                       unsigned int bat_size,
                                                       unsigned int vec_size);
template void reduce_mat_bp_host<REDUCE_MATRIX_OP_MEAN, false>(const float* A,
                                                        float*       A_grd,
                                                        const float* B,
                                                        const float* B_grd,
                                                        unsigned int bat_size,
                                                        unsigned int vec_size);
template void reduce_mat_bp_host<REDUCE_MATRIX_OP_MIN, false>(const float* A,
                                                       float*       A_grd,
                                                       const float* B,
                                                       const float* B_grd,
                                                       unsigned int bat_size,
                                                       unsigned int vec_size);
template void reduce_mat_bp_host<REDUCE_MATRIX_OP_MAX, false>(const float* A,
                                                       float*       A_grd,
                                                       const float* B,
                                                       const float* B_grd,
                                                       unsigned int bat_size,
                                                       unsigned int vec_size);


template void reduce_mat_bp_host<REDUCE_MATRIX_OP_SUM, true>(const float* A,
                                                              float*       A_grd,
                                                              const float* B,
                                                              const float* B_grd,
                                                              unsigned int bat_size,
                                                              unsigned int vec_size);
template void reduce_mat_bp_host<REDUCE_MATRIX_OP_MEAN, true>(const float* A,
                                                               float*       A_grd,
                                                               const float* B,
                                                               const float* B_grd,
                                                               unsigned int bat_size,
                                                               unsigned int vec_size);
template void reduce_mat_bp_host<REDUCE_MATRIX_OP_MIN, true>(const float* A,
                                                              float*       A_grd,
                                                              const float* B,
                                                              const float* B_grd,
                                                              unsigned int bat_size,
                                                              unsigned int vec_size);
template void reduce_mat_bp_host<REDUCE_MATRIX_OP_MAX, true>(const float* A,
                                                              float*       A_grd,
                                                              const float* B,
                                                              const float* B_grd,
                                                              unsigned int bat_size,
                                                              unsigned int vec_size);