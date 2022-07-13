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
// Created by Luecx on 11.07.2022.
//

#ifndef AD_LINEAR_H
#define AD_LINEAR_H

#include "../../operations/operations.h"
#include "../Node.h"

struct Linear : public Node<float> {
    Node<float>* m_previous;

    Tape<float>  m_weights;
    Tape<float>  m_bias;

    Linear(Node<float>* previous, int outSize)
        : m_previous(previous), Node(Dimension(outSize)),
          m_weights(Dimension((int) previous->getPartialDimension()[0], outSize)), m_bias(Dimension(outSize))
    {
        m_weights.values.randomiseGaussian(0, 2 / sqrt(previous->getPartialDimension()[0]));
        m_weights.values.gpuUpload();
    }

    virtual void forward() {
        affine<DEVICE>(m_previous->values, m_weights.values, m_bias.values, values);
    }
    virtual void backwards() {
        affine_bp<DEVICE>(m_previous->values,
                          m_previous->gradients,
                          m_weights.values,
                          m_weights.gradients,
                          m_bias.gradients,
                          gradients);
    }

    virtual void      clearWeightGradients() {
        m_weights.gradients.clear<DEVICE>();
        m_bias   .gradients.clear<DEVICE>();
    }

    std::vector<Tape<float>*> params() override {
        return {&m_weights, &m_bias};
    }
};

#endif    // AD_LINEAR_H
