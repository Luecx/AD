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

//
// Created by Luecx on 06.07.2022.
//

#ifndef AD_OPERATIONS_H
#define AD_OPERATIONS_H
// -------------------------------------- activation functions ---------------------------------------
// sigmoid
#include "sigmoid/sigmoid.h"
#include "sigmoid_bp/sigmoid_bp.h"
// relu
#include "relu/relu.h"
#include "relu_bp/relu_bp.h"
// clipped-relu
#include "clipped_relu/clipped_relu.h"
#include "clipped_relu_bp/clipped_relu_bp.h"
// swish
#include "swish/swish.h"
#include "swish_bp/swish_bp.h"
// clip
#include "clip/clip.h"
#include "clip_bp/clip_bp.h"

// ------------------------------------- element-wise functions --------------------------------------
// log
#include "log/log.h"
#include "log_bp/log_bp.h"
// exp
#include "exp/exp.h"
#include "exp_bp/exp_bp.h"

// add, sub, div and mul
#include "base_operator/base_operator.h"
#include "base_operator_bp/base_operator_bp.h"

// ---------------------------------------- special functions ----------------------------------------
// softmax
#include "softmax/softmax.h"
#include "softmax_bp/softmax_bp.h"

// -------------------------------------------- optimiser --------------------------------------------
// adam
#include "adam/adam.h"
// adam_w
#include "adam_w/adam_w.h"
// ranger
#include "ranger/ranger.h"
// r-adam
#include "radam/radam.h"

// ---------------------------------------------- misc -----------------------------------------------
// reduction
#include "reduce_mat/reduce_mat.h"
#include "reduce_mat_bp/reduce_mat_bp.h"
// selection (type of reduction)
#include "select/select.h"
#include "select_bp/select_bp.h"

// ------------------------------------ neural network operations ------------------------------------
// affine transformation
#include "affine/affine.h"
#include "affine_bp/affine_bp.h"

#endif    // AD_OPERATIONS_H
