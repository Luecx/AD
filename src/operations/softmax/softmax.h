//
// Created by Luecx on 02.07.2022.
//

#ifndef CUDAD_SRC_OPERATIONS_SOFTMAX_SOFTMAX_H_
#define CUDAD_SRC_OPERATIONS_SOFTMAX_SOFTMAX_H_

#include "../../array/Mode.h"
#include "../../tensor/Tensor.h"

// clang-format off
//void softmax_host(
//    const float* A,
//          float* B,
//    unsigned int m,
//    unsigned int n);
// clang-format on

// clang-format off
__global__ void softmax_kernel(
    const float* __restrict__ A,
          float* __restrict__ B,
    unsigned int m,
    unsigned int n);
// clang-format on

// clang-format off
template<Mode mode>
inline void softmax   (const Tensor<float> &A,
                             Tensor<float> &B){
    // clang-format on

    ASSERT(A.address<mode>(), "memory is not allocated for input tensor");
    ASSERT(B.address<mode>(), "memory is not allocated for output tensor");

    ASSERT(B.getDimension() == A.getDimension(),
           "size of input tensor and output tensor do not match");
    ASSERT(A.address<mode>(), "memory is not allocated for input tensor")
    ASSERT(B.address<mode>(), "memory is not allocated for output tensor");

    if (mode == DEVICE) {

        // clang-format off
        constexpr int block_size_m = 1;
        constexpr int block_size_n = 1024;
        dim3 block(block_size_n, block_size_m);
        dim3 grid (std::ceil((float)A.getDimension()[0] / block_size_n),1);
        softmax_kernel<<<grid, block>>>(
            A.gpuAddress(),
            B.gpuAddress(),
            A.getDimension()[0],
            A.getDimension()[1]);
        // clang-format on
    } else {
        // clang-format off
//        softmax_host(
//            A.cpuAddress(),
//            B.cpuAddress(),
//            A.getDimension()[0],
//            A.getDimension()[1]);
        // clang-format on
        ERROR(false, " not supported");
    }
}
// clang-format on

#endif    // CUDAD_SRC_OPERATIONS_SOFTMAX_SOFTMAX_H_
