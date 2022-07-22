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

#ifndef AD_CPUARRAY_H
#define AD_CPUARRAY_H

#include "Array.h"
#include <iostream>

template<typename Type>
struct CPUArray : Array<Type> {

    explicit CPUArray(ArraySizeType p_size) : Array<Type>(p_size) {
        this->m_data = new Type[p_size] {};
        clear();
    }
    virtual ~CPUArray() { delete[] this->m_data; }

    Type& operator()(ArraySizeType idx) { return this->m_data[idx]; }
    Type  operator()(ArraySizeType idx) const { return this->m_data[idx]; }
    Type& operator[](ArraySizeType idx) { return this->m_data[idx]; }
    Type  operator[](ArraySizeType idx) const { return this->m_data[idx]; }

    void  copyFrom(const CPUArray<Type>& other) {
//         ASSERT(other.size() == this->size(), "invalid dimension of second array");
         memcpy(this->m_data, other.m_data, std::min(this->size(), other.size()) * sizeof(Type));
    }
    void clear() { memset(this->m_data, 0, sizeof(Type) * this->size()); }
};

#endif    // AD_CPUARRAY_H
