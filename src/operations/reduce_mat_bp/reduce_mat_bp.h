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

#ifndef CUDATEST1_SRC_OPERATIONS_REDUCE_MAT_BP_REDUCE_MAT_BP_H_
#define CUDATEST1_SRC_OPERATIONS_REDUCE_MAT_BP_REDUCE_MAT_BP_H_

#include "../../array/Mode.h"
#include "../../misc/config.h"
#include "../../tensor/Tensor.h"
#include "../reduce_mat/reduce_mat.h"

// clang-format off
template<ReduceMatrixOperation operation, bool across_batch>
void reduce_mat_bp_host(
    const float* A,
          float* A_grd,
    const float* B,
    const float* B_grd,
    unsigned int bat_size,
    unsigned int vec_size);
// clang-format on

// clang-format off
template<ReduceMatrixOperation operation, bool across_batch>
__global__ void reduce_mat_bp_kernel(
    const float* __restrict__ A,
          float* __restrict__ A_grd,
    const float* __restrict__ B,
    const float* __restrict__ B_grd,
    unsigned int          bat_size,
    unsigned int          vec_size);
// clang-format on
// clang-format off
template<Mode mode, ReduceMatrixOperation operation, bool across_batch>
inline void reduce_mat_bp(const Tensor<float> &A,
                                Tensor<float> &A_grd,
                          const Tensor<float> &B,
                          const Tensor<float> &B_grd){
    // clang-format on

    ASSERT(A_grd.rank() == 2, "input is not a matrix");
    if constexpr (across_batch){
        ASSERT(B.rank() == 1 || B.rank() == 2 && B.getDimension()[0] == 1, "output has to be either a vector or matrix with dimension m == 1");
    }else{
        ASSERT(B.rank() == 1 || B.rank() == 2 && B.getDimension()[1] == 1, "output has to be either a vector or matrix with dimension n == 1");
    }
    ASSERT(A_grd.address<mode>(), "memory is not allocated for input tensor");
    ASSERT(B_grd.address<mode>(), "memory is not allocated for output tensor");

    if (mode == DEVICE) {

        constexpr int block_size_x = 16;
        constexpr int block_size_y = 16;
        dim3          block(block_size_x, block_size_y);
        dim3          grid(std::ceil((float) A_grd.getDimension()[1] / block_size_x),
                           std::ceil((float) A_grd.getDimension()[0] / block_size_y));
        // clang-format off
        reduce_mat_bp_kernel<operation, across_batch><<<grid, block>>>(
            A    .gpuAddress(),
            A_grd.gpuAddress(),
            B    .gpuAddress(),
            B_grd.gpuAddress(),
            A_grd.getDimension()[0],
            A_grd.getDimension()[1]);
        // clang-format on
    } else {
        // clang-format off
        reduce_mat_bp_host<operation, across_batch>(
            A    .cpuAddress(),
            A_grd.cpuAddress(),
            B    .cpuAddress(),
            B_grd.cpuAddress(),
            A_grd.getDimension()[0],
            A_grd.getDimension()[1]);
        // clang-format on
    }
}

// clang-format on
#endif    // CUDATEST1_SRC_OPERATIONS_REDUCE_MAT_BP_REDUCE_MAT_BP_H_
