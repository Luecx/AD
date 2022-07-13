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

// clang-format off
template<BaseOperation operation>
__global__ void base_operator_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
          float* __restrict__ C,
    unsigned int          size){
    // clang-format on

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= size)
        return;

    if constexpr (operation == BASE_OPERATOR_OP_ADD) {
        C[idx] = A[idx] + B[idx];
    } else if constexpr (operation == BASE_OPERATOR_OP_SUB) {
        C[idx] = A[idx] - B[idx];
    } else if constexpr (operation == BASE_OPERATOR_OP_DIV) {
        C[idx] = A[idx] / B[idx];
    } else if constexpr (operation == BASE_OPERATOR_OP_MUL) {
        C[idx] = A[idx] * B[idx];
    }
}

template void __global__ base_operator_kernel<BASE_OPERATOR_OP_ADD>(const float* __restrict__ A,
                                                                    const float* __restrict__ B,
                                                                    float* __restrict__ C,
                                                                    unsigned int size);
template void __global__ base_operator_kernel<BASE_OPERATOR_OP_SUB>(const float* __restrict__ A,
                                                                    const float* __restrict__ B,
                                                                    float* __restrict__ C,
                                                                    unsigned int size);
template void __global__ base_operator_kernel<BASE_OPERATOR_OP_DIV>(const float* __restrict__ A,
                                                                    const float* __restrict__ B,
                                                                    float* __restrict__ C,
                                                                    unsigned int size);
template void __global__ base_operator_kernel<BASE_OPERATOR_OP_MUL>(const float* __restrict__ A,
                                                                    const float* __restrict__ B,
                                                                    float* __restrict__ C,
                                                                    unsigned int size);