//
// Created by Luecx on 02.07.2022.
//

#ifndef CUDAD_SRC_OPERATIONS_SOFTMAX_BP_SOFTMAX_BP_H_
#define CUDAD_SRC_OPERATIONS_SOFTMAX_BP_SOFTMAX_BP_H_



// clang-format off
//void softmax_host(
//    const float* A,
//          float* B,
//    unsigned int m,
//    unsigned int n);

#include "../../array/Mode.h"
#include "../../tensor/Tensor.h"

__global__ void softmax_bp_kernel(
          float* __restrict__ A_grd,
    const float* __restrict__ B,
    const float* __restrict__ B_grd,
    unsigned int m,
    unsigned int n);

template<Mode mode>
inline void softmax_bp(const Tensor<float> &A,
                             Tensor<float> &A_grd,
                       const Tensor<float> &B,
                       const Tensor<float> &B_grd){

    ASSERT(A.address<mode>(), "memory is not allocated for input tensor");
    ASSERT(B.address<mode>(), "memory is not allocated for output tensor");

    ASSERT(B.getDimension() == A.getDimension(),
           "size of input tensor and output tensor do not match");
    ASSERT(A_grd.getDimension() == A.getDimension(),
           "size of input gradient and input tensor do not match");
    ASSERT(B_grd.getDimension() == B.getDimension(),
           "size of input gradient and input tensor do not match");

    ASSERT(A.address<mode>(), "memory is not allocated for input tensor")
    ASSERT(B.address<mode>(), "memory is not allocated for output tensor");
    ASSERT(A_grd.address<mode>(), "memory is not allocated for input gradient tensor")
    ASSERT(B_grd.address<mode>(), "memory is not allocated for output gradient tensor");

    if(mode == DEVICE){

        constexpr int block_size_m = 16;
        constexpr int block_size_n = 16;
        dim3 block(block_size_n, block_size_m);
        dim3 grid (std::ceil((float)A_grd.getDimension()[1] / block_size_n),std::ceil((float)A_grd.getDimension()[0] / block_size_m));
        softmax_bp_kernel<<<grid, block>>>(
            A_grd.gpuAddress(),
            B.gpuAddress(),
            B_grd.gpuAddress(),
            A_grd.getDimension()[0],
            A_grd.getDimension()[1]);
    }else{
//        softmax_host(
//            A.cpuAddress(),
//            B.cpuAddress(),
//            A.m,
//            A.n);
        ERROR(false, "operation not supported");
    }
}
// clang-format on

#endif    // CUDAD_SRC_OPERATIONS_SOFTMAX_BP_SOFTMAX_BP_H_
