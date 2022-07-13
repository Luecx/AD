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
// Created by Luecx on 05.07.2022.
//

#ifndef AD_TENSORDESCRIPTOR_H
#define AD_TENSORDESCRIPTOR_H

#include "../misc/config.h"
#include "Dimension.h"

#include <iostream>

template<typename Type>
struct TensorDescriptorObject {
    cudnnTensorDescriptor_t descriptor = nullptr;

    TensorDescriptorObject(Dimension dimension) {

        CUDA_ASSERT(cudnnCreateTensorDescriptor(&descriptor));

        auto tensor_format = CUDNN_TENSOR_NCHW;
        auto tensor_type   = CUDNN_DATA_FLOAT;

        if constexpr (std::is_same<double, Type>::value){
            tensor_type = CUDNN_DATA_DOUBLE;
        }else if constexpr (std::is_same<float, Type>::value){
            tensor_type = CUDNN_DATA_FLOAT;
        }else if constexpr (std::is_same<int, Type>::value){
            tensor_type = CUDNN_DATA_INT32;
        }else if constexpr (std::is_same<int8_t , Type>::value){
            tensor_type = CUDNN_DATA_INT8;
        }else if constexpr (std::is_same<int64_t , Type>::value){
            tensor_type = CUDNN_DATA_INT64;
        }else if constexpr (std::is_same<bool , Type>::value){
            tensor_type = CUDNN_DATA_BOOLEAN;
        }else{
            ERROR(false, "data type not supported for tensor");
        }

        if (dimension.rank() == 1) {
            cudnnSetTensor4dDescriptor(descriptor,
                                       tensor_format,
                                       tensor_type,
                                       1,
                                       1,
                                       1,
                                       dimension[0]);
        } else if (dimension.rank() == 2) {
            cudnnSetTensor4dDescriptor(descriptor,
                                       tensor_format,
                                       tensor_type,
                                       1,
                                       1,
                                       dimension[0],
                                       dimension[1]);
        } else if (dimension.rank() == 3) {
            cudnnSetTensor4dDescriptor(descriptor,
                                       tensor_format,
                                       tensor_type,
                                       1,
                                       dimension[0],
                                       dimension[1],
                                       dimension[2]);
        } else if (dimension.rank() == 4) {
            cudnnSetTensor4dDescriptor(descriptor,
                                       tensor_format,
                                       tensor_type,
                                       dimension[0],
                                       dimension[1],
                                       dimension[2],
                                       dimension[3]);
        } else {
            ERROR(false, "tensor of higher order are not yet supported");
        }
    }

    virtual ~TensorDescriptorObject() {
        cudnnDestroyTensorDescriptor(descriptor);
    }
};

#endif    // AD_TENSORDESCRIPTOR_H
