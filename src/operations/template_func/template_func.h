/**
    AD is a general CUDA neural network framework.
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

#ifndef AD_TEMPLATE_FUNC_H
#define AD_TEMPLATE_FUNC_H

#include "../../array/Mode.h"
#include "../../misc/config.h"
#include "../../tensor/Tensor.h"

#include <iostream>

// clang-format off
void template_func_host(
    const float* A,
          float* B,
    unsigned int size);

__global__ void template_func_kernel(
    const float* __restrict__ A,
          float* __restrict__ B,
    unsigned int size);
// clang-format on

template<Mode mode>
// clang-format off
inline void template_func(const Tensor<float> &A,
                                Tensor<float> &B){
    // clang-format on
    ASSERT(A.getDimension() == B.getDimension(), "Dimension of Tensor A and B do not match");

    if (mode == DEVICE) {

        ASSERT(A.gpuAddress(), "gpu not allocated for Tensor A");
        ASSERT(B.gpuAddress(), "gpu not allocated for Tensor B");

        constexpr int block_size = 1024;
        dim3          block(block_size);
        dim3          grid(std::ceil((float) A.size() / block_size));
        // clang-format off
        template_func_kernel<<<grid, block>>>(
            A.gpuAddress(),
            B.gpuAddress(),
            A.size());
        // clang-format on
    } else {
        ASSERT(A.cpuAddress(), "cpu not allocated for Tensor A");
        ASSERT(B.cpuAddress(), "cpu not allocated for Tensor B");
        // clang-format off
        template_func_host(
            A.cpuAddress(),
            B.cpuAddress(),
            A.size());
        // clang-format on
    }
}

#endif    // AD_TEMPLATE_FUNC_H
