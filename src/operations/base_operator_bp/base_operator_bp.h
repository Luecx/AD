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

#ifndef CUDATEST1_SRC_OPERATIONS_BASE_OPERATOR_BP_BASE_OPERATOR_BP_H_
#define CUDATEST1_SRC_OPERATIONS_BASE_OPERATOR_BP_BASE_OPERATOR_BP_H_

#include "../../array/Mode.h"
#include "../../misc/config.h"
#include "../../tensor/Tensor.h"
#include "../base_operator/base_operator.h"

#include <iostream>

// clang-format off
template<BaseOperation operation>
void base_operator_bp_host(
    const float* A,
          float* A_grd,
    const float* B,
          float* B_grd,
    const float* C_grd,
    unsigned int size);
// clang-format on

// clang-format off
template<BaseOperation operation>
__global__ void base_operator_bp_kernel(
    const float* __restrict__ A,
          float* __restrict__ A_grd,
    const float* __restrict__ B,
          float* __restrict__ B_grd,
    const float* __restrict__ C_grd,
    unsigned int              size);
// clang-format on

// clang-format off
template<Mode mode, BaseOperation operation>
inline void base_operator_bp(const Tensor<float> &A,
                                   Tensor<float> &A_grd,
                             const Tensor<float> &B,
                                   Tensor<float> &B_grd,
                             const Tensor<float> &C_grd) {
    // clang-format on

    // clang-format off
    ASSERT(A    .getDimension() == C_grd.getDimension(), "first input tensor and output gradient tensor must have same shape");
    ASSERT(A_grd.getDimension() == C_grd.getDimension(), "first input gradient tensor and output gradient tensor must have same shape");
    ASSERT(B    .getDimension() == C_grd.getDimension(), "second input tensor and output gradient tensor must have same shape");
    ASSERT(B_grd.getDimension() == C_grd.getDimension(), "second input gradient tensor and output gradient tensor must have same shape");
    // clang-format on

    // clang-format off
    ASSERT(A    .address<mode>(), "memory is not allocated for first input tensor");
    ASSERT(B    .address<mode>(), "memory is not allocated for second input tensor");
    ASSERT(A_grd.address<mode>(), "memory is not allocated for first input gradient tensor");
    ASSERT(B_grd.address<mode>(), "memory is not allocated for second input gradient tensor");
    ASSERT(C_grd.address<mode>(), "memory is not allocated for output gradient tensor");
    // clang-format on
    if (mode == DEVICE) {

        constexpr int block_size = 1024;
        dim3          block(block_size);
        dim3          grid(std::ceil((float) A.getDimension().size() / block_size));
        // clang-format off
        base_operator_bp_kernel<operation><<<grid, block>>>(
            A    .gpuAddress(),
            A_grd.gpuAddress(),
            B    .gpuAddress(),
            B_grd.gpuAddress(),
            C_grd.gpuAddress(),
            A.size());
        // clang-format on
    } else {
        // clang-format off
        base_operator_bp_host<operation>(
            A    .cpuAddress(),
            A_grd.cpuAddress(),
            B    .cpuAddress(),
            B_grd.cpuAddress(),
            C_grd.cpuAddress(),
            A.size());
        // clang-format on
    }
}

// clang-format on
#endif    // CUDATEST1_SRC_OPERATIONS_BASE_OPERATOR_BP_BASE_OPERATOR_BP_H_
