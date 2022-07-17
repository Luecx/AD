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

#ifndef CUDATEST1_SRC_OPERATIONS_BASE_OPERATOR_BASE_OPERATOR_H_
#define CUDATEST1_SRC_OPERATIONS_BASE_OPERATOR_BASE_OPERATOR_H_

#include "../../array/Mode.h"
#include "../../tensor/Tensor.h"

#include <iostream>

enum BaseOperation{
    BASE_OPERATOR_OP_ADD,
    BASE_OPERATOR_OP_SUB,
    BASE_OPERATOR_OP_DIV,
    BASE_OPERATOR_OP_MUL,
    BASE_OPERATOR_OP_MIN,
    BASE_OPERATOR_OP_MAX,
};

// clang-format off
template<BaseOperation operation>
void base_operator_host(
    const float* A,
    const float* B,
          float* C,
    unsigned int size);
// clang-format on

// clang-format off
template<BaseOperation operation>
__global__ void base_operator_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
          float* __restrict__ C,
    unsigned int              size);
// clang-format on

// clang-format off
template<Mode mode, BaseOperation operation>
inline void base_operator(const Tensor<float> &A,
                          const Tensor<float> &B,
                                Tensor<float> &C){
    // clang-format on

    ASSERT(A.getDimension() == C.getDimension(), "first input and output must have same shape")
    ASSERT(B.getDimension() == C.getDimension(), "second input and output must have same shape");

    ASSERT(A.address<mode>(), "memory is not allocated for first input tensor")
    ASSERT(B.address<mode>(), "memory is not allocated for second input tensor");
    ASSERT(C.address<mode>(), "memory is not allocated for output tensor");

    if (mode == DEVICE) {

        constexpr int block_size = 1024;
        dim3          block(block_size);
        dim3          grid(std::ceil((float) A.getDimension().size() / block_size));
        // clang-format off
        base_operator_kernel<operation><<<grid, block>>>(
            A.gpuAddress(),
            B.gpuAddress(),
            C.gpuAddress(),
            A.size());
        // clang-format on
    } else {
        // clang-format off
        base_operator_host<operation>(
            A.cpuAddress(),
            B.cpuAddress(),
            C.cpuAddress(),
            A.size());
        // clang-format on
    }
}

// clang-format on
#endif    // CUDATEST1_SRC_OPERATIONS_BASE_OPERATOR_BASE_OPERATOR_H_
