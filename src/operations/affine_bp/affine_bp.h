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

#ifndef AD_AFFINE_BP_H
#define AD_AFFINE_BP_H

#include "../../array/Mode.h"
#include "../../misc/config.h"
#include "../../tensor/Tensor.h"

#include <iostream>

// clang-format off
void affine_bp_host(
    const float* A,
          float* B,
    unsigned int size);

void affine_bp_kernel(const Tensor<float> &inp,
                            Tensor<float> &inp_grd,
                      const Tensor<float> &wgt,
                            Tensor<float> &wgt_grd,
                            Tensor<float> &bia_grd,
                      const Tensor<float> &out_grd);
// clang-format on

/**
 *
 * computes the gradients for the input (inp_grd) the weights (wgt_grd) and the bias (bia_grd) based
 * off the gradients from the output of the affine transformation. To allow weight sharing, the
 * inp_grd will be set whereas the gradients for the weights and biases is incremented.
 *
 * @tparam mode
 * @param inp
 * @param inp_grd
 * @param wgt
 * @param wgt_grd
 * @param bia_grd
 * @param out_grd
 */
template<Mode mode>
// clang-format off
inline void affine_bp(const Tensor<float> &inp,
                            Tensor<float> &inp_grd,
                      const Tensor<float> &wgt,
                            Tensor<float> &wgt_grd,
                            Tensor<float> &bia_grd,
                      const Tensor<float> &out_grd){
    // clang-format on

    ASSERT(inp.rank() == 2, "rank of input is not equal to 2");
    ASSERT(wgt.rank() == 2, "rank of weights is not equal to 2");
    ASSERT(bia_grd.rank() == 1, "rank of bias-gradient is not equal to 1");
    ASSERT(out_grd.rank() == 2, "rank of output-gradient is not equal to 2");

    ASSERT(inp.getDimension() == inp_grd.getDimension(), "input and input gradients to not match");
    ASSERT(wgt.getDimension() == wgt_grd.getDimension(), "weight and weight gradients to not match");

    ASSERT(inp.getDimension()[0] == out_grd.getDimension()[0], "first dimension of input and output do not match");
    ASSERT(inp.getDimension()[1] == wgt.getDimension()[0], "first dimension of weights must match the second dimension of the input");
    ASSERT(bia_grd.getDimension()[0] == out_grd.getDimension()[1], "first dimension of bias must match the second dimension of the output");
    ASSERT(out_grd.getDimension()[1] == wgt.getDimension()[1], "second dimension of the output must match the second dimension of the weights");

    if (mode == DEVICE) {

        affine_bp_kernel(inp, inp_grd, wgt, wgt_grd, bia_grd, out_grd);
        // clang-format on
    } else {
//        ASSERT(A.cpuAddress(), "cpu not allocated for Tensor A");
//        ASSERT(B.cpuAddress(), "cpu not allocated for Tensor B");
//        // clang-format off
//        affine_bp_host(
//            A.cpuAddress(),
//            B.cpuAddress(),
//            A.size());
//        // clang-format on
    }
}

#endif    // AD_AFFINE_BP_H
