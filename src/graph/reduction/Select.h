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

#ifndef AD_SELECT_H
#define AD_SELECT_H

#include "../../operations/select/select.h"
#include "../Node.h"
struct Select : public Node<float>{

    Node<float>* input{};
    Node<int>* indices{};

    Select(Node<float>* input, Node<int>* indices)
        : Node(indices->getPartialDimension()), input(input), indices(indices) {}

    virtual void forward() {
        select<DEVICE>(input->values, indices->values, this->values);
    }
    virtual void backwards() {
        select_bp<DEVICE>(input->gradients, indices->values, this->gradients);}

    virtual void      clearWeightGradients() {}
};

#endif    // AD_SELECT_H
