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

#ifndef AD_NODE_H
#define AD_NODE_H

#include "../tensor/Dimension.h"
#include "../tensor/Tensor.h"
#include "Tape.h"

enum NodeIType { NIT_FLOAT, NIT_BINARY, NIT_INTEGER, NIT_N_TYPES = 3 };

// get the node type from a given templated type
template<typename Type>
inline NodeIType deriveNodeType() {
    if constexpr (std::is_same<Type, float>::value) {
        return NIT_FLOAT;
    } else if constexpr (std::is_same<Type, int>::value) {
        return NIT_INTEGER;
    } else if constexpr (std::is_same<Type, bool>::value) {
        return NIT_BINARY;
    } else {
        ERROR(false, "Cannot derive node type from given type");
    }
}

struct NodeInterface {
    public:
    // feed the data forward
    virtual void forward() = 0;
    // feed the data backwards
    virtual void backwards() = 0;

    // reset the gradients of the layer
    virtual void clearGradients() = 0;
    // sets the batch size which should reallocate the output value and gradient tensor
    virtual void setBatchSize(int batch_size) = 0;

    // returns the total dimension including the batch size
    virtual Dimension getDimension() = 0;
    // returns the dimension without considering batch size
    virtual Dimension getPartialDimension() = 0;

    // returns what type of data is stored in this node
    virtual NodeIType getNodeType() = 0;

    // upload / download the values / gradients to / from the gpu
    virtual void uploadValues()      = 0;
    virtual void downloadValues()    = 0;
    virtual void uploadGradients()   = 0;
    virtual void downloadGradients() = 0;

    // get a list of all tunable params in this layer
    virtual std::vector<Tape<float>*> params() = 0;
};

template<typename Type>
struct Node : public Tape<Type>, NodeInterface {

    const Dimension m_partial_dimension;

    Node(Dimension p_partial_dimension)
        : m_partial_dimension(p_partial_dimension), Tape<Type> {Dimension(1) + p_partial_dimension} {}

    private:
    virtual void clearWeightGradients() = 0;

    public:
    virtual void forward()   = 0;
    virtual void backwards() = 0;

    // clearing gradients
    void clearGradients() {
        clearWeightGradients();
        this->gradients.template clear<DEVICE>();
        this->gradients.template clear<HOST>();
    }

    // setting batch size
    void setBatchSize(int batch_size) {
        Tape<Type>::operator=(Tape<Type> {Dimension(batch_size) + getPartialDimension()});
    }

    // returns the dimension without considering batch size
    Dimension getPartialDimension() { return m_partial_dimension; }

    // returns the total dimension including the batch size
    Dimension getDimension() override { return this->values.getDimension(); }

    // returns what type of data is stored in this node
    NodeIType getNodeType() override { return deriveNodeType<Type>(); }

    // upload / download the values / gradients to / from the gpu
    void uploadValues() override { this->values.gpuUpload(); }
    void downloadValues() override { this->values.gpuDownload(); }
    void uploadGradients() override { this->gradients.gpuUpload(); }
    void downloadGradients() override { this->gradients.gpuDownload(); }

    // get a list of all tunable params in this layer
    virtual std::vector<Tape<float>*> params() override { return std::vector<Tape<float>*>(); }
};

#endif    // AD_NODE_H
