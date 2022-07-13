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

#ifndef AD_TEMPLATE_KERNEL_H
#define AD_TEMPLATE_KERNEL_H

#include "affine.h"

// clang-format off
void affine_kernel(
    const Tensor<float> &inp,
    const Tensor<float> &wgt,
    const Tensor<float> &bia,
          Tensor<float> &out){
    // clang-format on

    ASSERT(inp.gpuAddress(), "gpu not allocated for input");
    ASSERT(wgt.gpuAddress(), "gpu not allocated for weights");
    ASSERT(bia.gpuAddress(), "gpu not allocated for bias");
    ASSERT(out.gpuAddress(), "gpu not allocated for output");

    const int out_size = out.getDimension()[1];
    const int inp_size = inp.getDimension()[1];
    const int bat_size = inp.getDimension()[0];

    const int m        = out_size;
    const int n        = bat_size;
    const int k        = inp_size;

    int       lda      = m;
    int       ldb      = k;
    int       ldc      = m;

    auto      trans_a  = CUBLAS_OP_N;
    auto      trans_b  = CUBLAS_OP_N;

    float     alpha    = 1;
    float     beta     = 0;

    // clang-format off
    cublasSgemm(CUBLAS_HANDLE, trans_a, trans_b,
            m, n, k, &alpha,
            wgt.gpuAddress(), lda,
            inp.gpuAddress(), ldb, &beta,
            out.gpuAddress(), ldc);
    beta = 1;
    cudnnAddTensor(CUDNN_HANDLE,
                   &beta , bia.getDescriptor()->descriptor, bia.gpuAddress(),
                   &alpha, out.getDescriptor()->descriptor, out.gpuAddress());
    // clang-format on

    CUDA_ASSERT(cudaPeekAtLastError());
}
// clang-format on
#endif    // AD_TEMPLATE_KERNEL_H
