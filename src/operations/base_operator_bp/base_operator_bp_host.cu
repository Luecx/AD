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

#include "base_operator_bp.h"

#include <iostream>

// clang-format off
template<BaseOperation operation>
void base_operator_bp_host(
    const float* A,
          float* A_grd,
    const float* B,
          float* B_grd,
    const float* C_grd,
    unsigned int size){
    // clang-format on

    for (int idx = 0; idx < size; idx++) {
        if constexpr (operation == BASE_OPERATOR_OP_ADD) {
            A_grd[idx] += C_grd[idx];
            B_grd[idx] += C_grd[idx];
        } else if constexpr (operation == BASE_OPERATOR_OP_SUB) {
            A_grd[idx] += C_grd[idx];
            B_grd[idx] -= C_grd[idx];
        } else if constexpr (operation == BASE_OPERATOR_OP_DIV) {
            A_grd[idx] += C_grd[idx] / B[idx];
            B_grd[idx] -= C_grd[idx] * A[idx] / (B[idx] * B[idx]);
        } else if constexpr (operation == BASE_OPERATOR_OP_MUL) {
            A_grd[idx] += C_grd[idx] * B[idx];
            B_grd[idx] += C_grd[idx] * A[idx];
        }
    }
}

template void base_operator_bp_host<BASE_OPERATOR_OP_ADD>(const float* A,
                                                          float*       A_grd,
                                                          const float* B,
                                                          float*       B_grd,
                                                          const float* C_grd,
                                                          unsigned int size);
template void base_operator_bp_host<BASE_OPERATOR_OP_SUB>(const float* A,
                                                          float*       A_grd,
                                                          const float* B,
                                                          float*       B_grd,
                                                          const float* C_grd,
                                                          unsigned int size);
template void base_operator_bp_host<BASE_OPERATOR_OP_DIV>(const float* A,
                                                          float*       A_grd,
                                                          const float* B,
                                                          float*       B_grd,
                                                          const float* C_grd,
                                                          unsigned int size);
template void base_operator_bp_host<BASE_OPERATOR_OP_MUL>(const float* A,
                                                          float*       A_grd,
                                                          const float* B,
                                                          float*       B_grd,
                                                          const float* C_grd,
                                                          unsigned int size);