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

#include "Softmax.h"
#include "../../operations/operations.h"

Softmax::Softmax(Node<float>* previousNode)
    : previous_node(previousNode), Node<float> {previousNode->getPartialDimension()} {}

void Softmax::clearWeightGradients() {}

void Softmax::forward() { softmax<DEVICE>(previous_node->values, this->values); }

void Softmax::backwards() {
    softmax_bp<DEVICE>(previous_node->values,
                       previous_node->gradients,
                       this->values,
                       this->gradients);
}
