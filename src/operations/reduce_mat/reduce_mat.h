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

#ifndef CUDATEST1_SRC_OPERATIONS_REDUCE_MAT_REDUCE_MAT_H_
#define CUDATEST1_SRC_OPERATIONS_REDUCE_MAT_REDUCE_MAT_H_

#include "../../array/Mode.h"
#include "../../tensor/Tensor.h"

#include <iostream>

enum ReduceMatrixOperation{
    REDUCE_MATRIX_OP_SUM,
    REDUCE_MATRIX_OP_MEAN,
    REDUCE_MATRIX_OP_MIN,
    REDUCE_MATRIX_OP_MAX
};

// clang-format off
template<ReduceMatrixOperation operation, bool across_batch>
void reduce_mat_host(
    const float* A,
          float* B,
    unsigned int          bat_size,
    unsigned int          vec_size);
// clang-format on

// clang-format off
template<ReduceMatrixOperation operation, bool across_batch>
__global__ void reduce_mat_kernel(
    const float* __restrict__ A,
          float* __restrict__ B,
    unsigned int          bat_size,
    unsigned int          vec_size);
// clang-format on
// clang-format off
template<Mode mode, ReduceMatrixOperation operation, bool across_batch>
inline void reduce_mat(const Tensor<float> &A,
                             Tensor<float> &B){
    // clang-format on

    ASSERT(A.rank() == 2, "input is not a matrix");
    if constexpr (across_batch){
        ASSERT(B.rank() == 1 || B.rank() == 2 && B.getDimension()[0] == 1, "output has to be either a vector or matrix with dimension m == 1");
    }else{
        ASSERT(B.rank() == 1 || B.rank() == 2 && B.getDimension()[1] == 1, "output has to be either a vector or matrix with dimension n == 1");
    }

    ASSERT(A.address<mode>(), "memory is not allocated for input tensor");
    ASSERT(B.address<mode>(), "memory is not allocated for output tensor");

    if (mode == DEVICE) {

        int threads = across_batch ? A.getDimension()[1] : A.getDimension()[0];

        constexpr int block_size = 1024;
        dim3          block(1, block_size);
        dim3          grid(1, std::ceil((float) threads / block_size));
        // clang-format off
        reduce_mat_kernel<operation, across_batch><<<grid, block>>>(
            A.gpuAddress(),
            B.gpuAddress(),
            A.getDimension()[0],
            A.getDimension()[1]);
        // clang-format on
    } else {
        // clang-format off
        reduce_mat_host<operation, across_batch>(
            A.cpuAddress(),
            B.cpuAddress(),
            A.getDimension()[0],
            A.getDimension()[1]);
        // clang-format on
    }
}

// clang-format on
#endif    // CUDATEST1_SRC_OPERATIONS_REDUCE_MAT_REDUCE_MAT_H_
