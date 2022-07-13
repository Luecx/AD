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

#include "Array.h"
#include "CPUArray.h"

#ifndef AD_GPUARRAY_H
#define AD_GPUARRAY_H

template<typename Type>
struct GPUArray : Array<Type> {
    explicit GPUArray(ArraySizeType p_size) : Array<Type>(p_size) {
        CUDA_ASSERT(cudaMalloc(&this->m_data, this->size() * sizeof(Type)));
        clear();
    }
    virtual ~GPUArray() { CUDA_ASSERT(cudaFree(this->m_data)); }

    void upload(CPUArray<Type>& cpu_array) {
        CUDA_ASSERT(cudaMemcpy(this->m_data,
                               cpu_array.m_data,
                               this->m_size * sizeof(Type),
                               cudaMemcpyHostToDevice));
    }
    void download(CPUArray<Type>& cpu_array) {
        CUDA_ASSERT(cudaMemcpy(cpu_array.m_data,
                               this->m_data,
                               this->m_size * sizeof(Type),
                               cudaMemcpyDeviceToHost));
    }

    void copyFrom(const GPUArray<Type>& other) {
        ASSERT(other.size() == this->size(), "invalid dimension of second array");
        cudaMemcpy(this->m_data, other.m_data, this->size() * sizeof(Type), cudaMemcpyDeviceToDevice);
    }
    void clear() { cudaMemset(this->m_data, 0, sizeof(Type) * this->size()); }
};

#endif    // AD_GPUARRAY_H
