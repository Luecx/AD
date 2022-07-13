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

#ifndef CUDATEST1_SRC_OPERATIONS_SELECT_SELECT_H_
#define CUDATEST1_SRC_OPERATIONS_SELECT_SELECT_H_

#include "../../array/Mode.h"
#include "../../tensor/Tensor.h"

#include <iostream>

// clang-format off
void select_host(
    const float* A,
    const int  * index,
          float* B,
    unsigned int m,
    unsigned int n);
// clang-format on
// clang-format off
__global__ void select_kernel(
    const float* __restrict__ A,
    const int  * __restrict__ index,
          float* __restrict__ B,
    unsigned int m,
    unsigned int n);
// clang-format on
// clang-format off
template<Mode mode>
inline void select   (const Tensor<float> &A,
                      const Tensor<int> &index,
                            Tensor<float> &B){
    // clang-format on

    ASSERT(A.address    <mode>(), "memory is not allocated for input tensor");
    ASSERT(index.address<mode>(), "memory is not allocated for index tensor");
    ASSERT(B.address    <mode>(), "memory is not allocated for output tensor");

    ASSERT(index.rank() == 1 || index.rank() == 2 && index.getDimension()[1] == 1,
           "rank of index tensor must either be 1 or have a second dimension of 1")
    ASSERT(B.rank() == 1 || B.rank() == 2 && B.getDimension()[1] == 1,
           "rank of output tensor must either be 1 or have a second dimension of 1")
    ASSERT(A.getDimension().rank() == 2, "rank of input and output tensor must be 2");
    ASSERT(A.getDimension()[0] == B.getDimension()[0], "input and output tensor must have the same first dimension");
    ASSERT(index.getDimension() == B.getDimension(), "index and output tensor must have same shape");

    if (mode == DEVICE) {

        constexpr int block_size = 1024;
        dim3             block(1, block_size);
        dim3             grid(1, std::ceil((float) A.getDimension()[0] / block_size));
        // clang-format off
        select_kernel<<<grid, block>>>(
            A.gpuAddress(),
            index.gpuAddress(),
            B.gpuAddress(),
            A.getDimension()[0],
            A.getDimension()[1]);
        // clang-format on
    } else {
        // clang-format off
        select_host(
            A.cpuAddress(),
            index.cpuAddress(),
            B.cpuAddress(),
            A.getDimension()[0],
            A.getDimension()[1]);
        // clang-format on
    }
}

// clang-format on
#endif    // CUDATEST1_SRC_OPERATIONS_SELECT_SELECT_H_
