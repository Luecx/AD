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

#include <iostream>

// clang-format off
template<ReduceMatrixOperation operation, bool across_batch>
void reduce_mat_host(
    const float* A,
          float* B,
    unsigned int          bat_size,
    unsigned int          vec_size){
    // clang-format on

    for (int b = 0; b < (across_batch ? vec_size : bat_size); b++) {
        float temp;

        if constexpr (operation == REDUCE_MATRIX_OP_SUM ||
                      operation == REDUCE_MATRIX_OP_MEAN) {
            temp = 0;
        } else if constexpr (operation == REDUCE_MATRIX_OP_MIN ||
                             operation == REDUCE_MATRIX_OP_MAX) {
            if constexpr (across_batch){
                temp = A[INDEX_2D(vec_size, 0, b)];
            }else{
                temp = A[INDEX_2D(vec_size, 0, b)];
            }
        }

        for (int idx = 0; idx < (across_batch ? bat_size : vec_size); idx++) {

            int mat_index;
            if constexpr (across_batch){
                mat_index = INDEX_2D(vec_size, idx, b);
            }else{
                mat_index = INDEX_2D(vec_size, b, idx);
            }

            if constexpr (operation == REDUCE_MATRIX_OP_SUM) {
                temp += A[mat_index];
            } else if constexpr (operation == REDUCE_MATRIX_OP_MEAN) {
                temp += A[mat_index];
            } else if constexpr (operation == REDUCE_MATRIX_OP_MIN) {
                temp = std::min(A[mat_index], temp);
            } else if constexpr (operation == REDUCE_MATRIX_OP_MAX) {
                temp = std::max(A[mat_index], temp);
            }
        }

        if constexpr (operation == REDUCE_MATRIX_OP_SUM) {
            B[b] = temp;
        } else if constexpr (operation == REDUCE_MATRIX_OP_MEAN) {
            B[b] = temp / vec_size;
        } else if constexpr (operation == REDUCE_MATRIX_OP_MIN) {
            B[b] = temp;
        } else if constexpr (operation == REDUCE_MATRIX_OP_MAX) {
            B[b] = temp;
        }
    }
}

template void reduce_mat_host<REDUCE_MATRIX_OP_SUM, true>(const float* A,
                                                          float*       B,
                                                          unsigned int bat_size,
                                                          unsigned int vec_size);
template void reduce_mat_host<REDUCE_MATRIX_OP_MEAN, true>(const float* A,
                                                           float*       B,
                                                           unsigned int bat_size,
                                                           unsigned int vec_size);
template void reduce_mat_host<REDUCE_MATRIX_OP_MIN, true>(const float* A,
                                                          float*       B,
                                                          unsigned int bat_size,
                                                          unsigned int vec_size);
template void reduce_mat_host<REDUCE_MATRIX_OP_MAX, true>(const float* A,
                                                          float*       B,
                                                          unsigned int bat_size,
                                                          unsigned int vec_size);

template void reduce_mat_host<REDUCE_MATRIX_OP_SUM, false>(const float* A,
                                                           float*       B,
                                                           unsigned int bat_size,
                                                           unsigned int vec_size);
template void reduce_mat_host<REDUCE_MATRIX_OP_MEAN, false>(const float* A,
                                                            float*       B,
                                                            unsigned int bat_size,
                                                            unsigned int vec_size);
template void reduce_mat_host<REDUCE_MATRIX_OP_MIN, false>(const float* A,
                                                           float*       B,
                                                           unsigned int bat_size,
                                                           unsigned int vec_size);
template void reduce_mat_host<REDUCE_MATRIX_OP_MAX, false>(const float* A,
                                                           float*       B,
                                                           unsigned int bat_size,
                                                           unsigned int vec_size);