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

#ifndef CUDATEST1_SRC_OPERATIONS_SELECT_BP_SELECT_BP_H_
#define CUDATEST1_SRC_OPERATIONS_SELECT_BP_SELECT_BP_H_

#include "../../array/Mode.h"
#include "../../tensor/Tensor.h"

#include <iostream>

// clang-format off
void select_bp_host(
          float* A_grd,
    const int  * index,
    const float* B_grd,
    unsigned int m,
    unsigned int n);
// clang-format on
// clang-format off
__global__ void select_bp_kernel(
          float* __restrict__ A_grd,
    const int  * __restrict__ index,
    const float* __restrict__ B_grd,
    unsigned int m,
    unsigned int n);
// clang-format on
// clang-format off
template<Mode mode>
inline void select_bp(      Tensor<float> &A_grd,
                      const Tensor<int  > &index,
                      const Tensor<float> &B_grd){
    // clang-format on

    ASSERT(A_grd.address<mode>(), "memory is not allocated for input tensor");
    ASSERT(index.address<mode>(), "memory is not allocated for index tensor");
    ASSERT(B_grd.address<mode>(), "memory is not allocated for output tensor");

    ASSERT(index.rank() == 1 || index.rank() == 2 && index.getDimension()[1] == 1,
           "rank of index tensor must either be 1 or have a second dimension of 1")
    ASSERT(B_grd.rank() == 1 || B_grd.rank() == 2 && B_grd.getDimension()[1] == 1,
           "rank of output tensor must either be 1 or have a second dimension of 1")
    ASSERT(A_grd.getDimension().rank() == 2, "rank of input and output tensor must be 2");
    ASSERT(A_grd.getDimension()[0] == B_grd.getDimension()[0], "input and output tensor must have the same first dimension");
    ASSERT(index.getDimension() == B_grd.getDimension(), "index and output tensor must have same shape");

    if (mode == DEVICE) {

        constexpr int block_size = 1024;
        dim3             block(1, block_size);
        dim3             grid(1, std::ceil((float) A_grd.getDimension()[0] / block_size));
        // clang-format off
        select_bp_kernel<<<grid, block>>>(
            A_grd.gpuAddress(),
            index.gpuAddress(),
            B_grd.gpuAddress(),
            A_grd.getDimension()[0],
            A_grd.getDimension()[1]);
        // clang-format on
    } else {
        // clang-format off
        select_bp_host(
            A_grd.cpuAddress(),
            index.cpuAddress(),
            B_grd.cpuAddress(),
            A_grd.getDimension()[0],
            A_grd.getDimension()[1]);
        // clang-format on
    }
}

// clang-format on
#endif    // CUDATEST1_SRC_OPERATIONS_SELECT_BP_SELECT_BP_H_
