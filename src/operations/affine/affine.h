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

#ifndef AD_AFFINE_H
#define AD_AFFINE_H

#include "../../array/Mode.h"
#include "../../misc/config.h"
#include "../../tensor/Tensor.h"

#include <iostream>

// clang-format off
void affine_host(
    const float* A,
          float* B,
    unsigned int size);

void affine_kernel(
    const Tensor<float> &inp,
    const Tensor<float> &wgt,
    const Tensor<float> &bia,
          Tensor<float> &out);
// clang-format on

template<Mode mode>
// clang-format off
inline void affine(const Tensor<float> &inp,
                   const Tensor<float> &wgt,
                   const Tensor<float> &bia,
                         Tensor<float> &out){

    ASSERT(inp.rank() == 2, "rank of input is not equal to 2");
    ASSERT(wgt.rank() == 2, "rank of weights is not equal to 2");
    ASSERT(bia.rank() == 1, "rank of bias is not equal to 1");
    ASSERT(out.rank() == 2, "rank of output is not equal to 2");

    ASSERT(inp.getDimension()[0] == out.getDimension()[0], "first dimension of input and output do not match");
    ASSERT(inp.getDimension()[1] == wgt.getDimension()[0], "first dimension of weights must match the second dimension of the input");
    ASSERT(bia.getDimension()[0] == out.getDimension()[1], "first dimension of bias must match the second dimension of the output");
    ASSERT(out.getDimension()[1] == wgt.getDimension()[1], "second dimension of the output must match the second dimension of the weights");

    if (mode == DEVICE) {
        affine_kernel(inp, wgt, bia, out);
    } else {
//        ASSERT(A.cpuAddress(), "cpu not allocated for Tensor A");
//        ASSERT(B.cpuAddress(), "cpu not allocated for Tensor B");
//        // clang-format off
//        affine_host(
//            A.cpuAddress(),
//            B.cpuAddress(),
//            A.size());
//        // clang-format on
    }
}

#endif    // AD_AFFINE_H
