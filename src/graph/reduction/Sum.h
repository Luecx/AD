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
// Created by Luecx on 10.07.2022.
//

#ifndef AD_SUM_H
#define AD_SUM_H
#include "../../operations/reduce_mat/reduce_mat.h"
#include "../Node.h"
struct Sum : public Node<float> {

    Node<float>* input {};

    Sum(Node<float>* input)
        : Node<float> {Dimension(1)}, input(input) {}

    virtual void forward() {
        reduce_mat<DEVICE, REDUCE_MATRIX_OP_SUM, false>(input->values, this->values);
    }
    virtual void backwards() {
        reduce_mat_bp<DEVICE, REDUCE_MATRIX_OP_SUM, false>(input->values, input->gradients, this->values, this->gradients);
    }

    virtual void      clearWeightGradients() {}
};
#endif    // AD_MEAN_H
