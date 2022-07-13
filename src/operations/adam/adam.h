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


#ifndef CUDATEST1_SRC_OPERATIONS_ADAM_ADAM_H_
#define CUDATEST1_SRC_OPERATIONS_ADAM_ADAM_H_

#include "../../array/Mode.h"
#include "../../tensor/Tensor.h"

#include <iostream>

// clang-format off
__global__ void adam_kernel(
          float* __restrict__ values,
          float* __restrict__ gradients,
          float* __restrict__ first_moment,
          float* __restrict__ second_moment,
          int size,
          float alpha,
          float beta1,
          float beta2,
          float eps);
// clang-format on

// clang-format off
void adam_host(
          float* values,
          float* gradients,
          float* first_moment,
          float* second_moment,
          int size,
          float alpha,
          float beta1,
          float beta2,
          float eps);
// clang-format on

// clang-format off
template<Mode mode>
inline void adam(Tensor<float>& values,
                 Tensor<float>& gradients,
                 Tensor<float>& first_moment,
                 Tensor<float>& second_moment,
                 float alpha,
                 float beta1,
                 float beta2,
                 float eps){
    // clang-format on

    constexpr int block_size = 1024;

    ASSERT(values.getDimension() == gradients.getDimension(),
           "dimension of values and gradients does not match");
    ASSERT(values.getDimension() == first_moment.getDimension(),
           "dimension of values and first moment vector does not match")
    ASSERT(values.getDimension() == second_moment.getDimension(),
           "dimension of values and second moment vector does not match")

    ASSERT(values.address<mode>(), "memory for values is not allocated");
    ASSERT(gradients.address<mode>(), "memory for gradients is not allocated");
    ASSERT(first_moment.address<mode>(), "memory for first moment vector is not allocated");
    ASSERT(second_moment.address<mode>(), "memory for second moment vector is not allocated");

    if (mode == DEVICE) {
        dim3 block(block_size);
        dim3 grid(std::ceil((float) values.size() / block_size));
        // clang-format off
        adam_kernel<<<grid, block>>>(
            values          .gpuAddress(),
            gradients       .gpuAddress(),
            first_moment    .gpuAddress(),
            second_moment   .gpuAddress(),
            values.size(),
            alpha, beta1, beta2, eps);
        // clang-format on
    } else {
        // clang-format off
        adam_host(
            values          .cpuAddress(),
            gradients       .cpuAddress(),
            first_moment    .cpuAddress(),
            second_moment   .cpuAddress(),
            values.size(),
            alpha, beta1, beta2, eps);
        // clang-format on
    }
    //    cudaDeviceSynchronize();
}
// clang-format on

#endif    // CUDATEST1_SRC_OPERATIONS_ADAM_ADAM_H_
