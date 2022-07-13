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

#include "affine_bp.h"

#define MATRIX_INDEX(size_x, m, n) (m * size_x + n)

__global__ void reduce_add_matrix(
          float* __restrict__ vec_grd,
    const float* __restrict__ res_grd,
    int m,
    int n){
    // clang-format on

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx >= n || idy >= m)
        return;

    float res_grd_v = res_grd[MATRIX_INDEX(n, idy, idx)];

    if (res_grd_v == 0)
        return;

    atomicAdd(&vec_grd[idx], res_grd_v);
}

#undef MATRIX_INDEX



// clang-format off
void affine_bp_kernel(const Tensor<float> &inp,
                            Tensor<float> &inp_grd,
                      const Tensor<float> &wgt,
                            Tensor<float> &wgt_grd,
                            Tensor<float> &bia_grd,
                      const Tensor<float> &out_grd){
    // clang-format on

    ASSERT(inp.gpuAddress(), "gpu not allocated for input");
    ASSERT(inp_grd.gpuAddress(), "gpu not allocated for input gradient");
    ASSERT(wgt.gpuAddress(), "gpu not allocated for weight");
    ASSERT(wgt_grd.gpuAddress(), "gpu not allocated for weight gradient");
    ASSERT(bia_grd.gpuAddress(), "gpu not allocated for bias gradient");
    ASSERT(out_grd.gpuAddress(), "gpu not allocated for output");

    constexpr int block_size_x = 16;
    constexpr int block_size_y = 16;
    dim3 block(block_size_x, block_size_y);
    dim3 grid (std::ceil((float)out_grd.getDimension()[1] / block_size_x),
               std::ceil((float)out_grd.getDimension()[0] / block_size_y));

    reduce_add_matrix<<<grid, block>>>(bia_grd.gpuAddress(),
                                       out_grd.gpuAddress(),
                                       out_grd.getDimension()[0],
                                       out_grd.getDimension()[1]);

    float alpha = 1;
    float beta  = 1;

    cublasSgemm(CUBLAS_HANDLE,
                CUBLAS_OP_N,
                CUBLAS_OP_T,
                out_grd.getDimension()[1],
                inp_grd.getDimension()[1],
                inp_grd.getDimension()[0],
                &alpha,
                out_grd.gpuAddress(), out_grd.getDimension()[1],
                inp.gpuAddress()    , inp    .getDimension()[1],
                &beta,
                wgt_grd.gpuAddress(), wgt_grd.getDimension()[1]);

    CUDA_ASSERT(cudaPeekAtLastError());

    beta  = 0;
    cublasSgemm(CUBLAS_HANDLE,
                CUBLAS_OP_T,
                CUBLAS_OP_N,
                inp_grd.getDimension()[1],
                inp_grd.getDimension()[0],
                out_grd.getDimension()[1],
                &alpha,
                wgt.gpuAddress()    , wgt    .getDimension()[1],
                out_grd.gpuAddress(), out_grd.getDimension()[1],
                &beta,
                inp_grd.gpuAddress(), inp_grd.getDimension()[1]);

    CUDA_ASSERT(cudaPeekAtLastError());
}
// clang-format on
#endif    // AD_TEMPLATE_KERNEL_H
