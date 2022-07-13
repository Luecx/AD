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

#include "../../operations/operations.h"
#include "Sub.h"

Sub::Sub(Node<float>* prev_1, Node<float>* prev_2)
    : m_prev_1(prev_1), m_prev_2(prev_2), Node<float> {prev_1->getPartialDimension()} {}

void Sub::clearWeightGradients() {}

void Sub::forward() {
    base_operator<DEVICE, BASE_OPERATOR_OP_SUB>(m_prev_1->values, m_prev_2->values, values);
}

void Sub::backwards() {
    base_operator_bp<DEVICE, BASE_OPERATOR_OP_SUB>(m_prev_1->values,
                                                   m_prev_1->gradients,
                                                   m_prev_2->values,
                                                   m_prev_2->gradients,
                                                   gradients);
}
