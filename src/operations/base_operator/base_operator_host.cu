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

#include "base_operator.h"

#include <iostream>

// clang-format off
template<BaseOperation operation>
void base_operator_host(
    const float* A,
    const float* B,
          float* C,
    unsigned int size){
    // clang-format on

    for (int idx = 0; idx < size; idx++) {

        if constexpr (operation == BASE_OPERATOR_OP_ADD) {
            C[idx] = A[idx] + B[idx];
        } else if constexpr (operation == BASE_OPERATOR_OP_SUB) {
            C[idx] = A[idx] - B[idx];
        } else if constexpr (operation == BASE_OPERATOR_OP_DIV) {
            C[idx] = A[idx] / B[idx];
        } else if constexpr (operation == BASE_OPERATOR_OP_MUL) {
            C[idx] = A[idx] * B[idx];
        } else if constexpr (operation == BASE_OPERATOR_OP_MIN) {
            C[idx] = std::min(A[idx], B[idx]);
        } else if constexpr (operation == BASE_OPERATOR_OP_MAX) {
            C[idx] = std::max(A[idx], B[idx]);
        }
    }
}

template void base_operator_host<BASE_OPERATOR_OP_ADD>(const float* A,
                                                       const float* B,
                                                       float*       C,
                                                       unsigned int size);
template void base_operator_host<BASE_OPERATOR_OP_SUB>(const float* A,
                                                       const float* B,
                                                       float*       C,
                                                       unsigned int size);
template void base_operator_host<BASE_OPERATOR_OP_DIV>(const float* A,
                                                       const float* B,
                                                       float*       C,
                                                       unsigned int size);
template void base_operator_host<BASE_OPERATOR_OP_MUL>(const float* A,
                                                       const float* B,
                                                       float*       C,
                                                       unsigned int size);
template void base_operator_host<BASE_OPERATOR_OP_MIN>(const float* A,
                                                       const float* B,
                                                       float*       C,
                                                       unsigned int size);
template void base_operator_host<BASE_OPERATOR_OP_MAX>(const float* A,
                                                       const float* B,
                                                       float*       C,
                                                       unsigned int size);