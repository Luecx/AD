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

#ifndef AD_TENSOR_H
#define AD_TENSOR_H

#include "../array/Array.h"
#include "../array/SArray.h"
#include "../assert/Error.h"
#include "TensorDescriptor.h"

#include <memory>
#include <iostream>
#include <type_traits>

template<typename Type = float>
class Tensor : public SArray<Type> {
    // typedef for tensor descriptor
    using TensorDescriptorO = TensorDescriptorObject<Type>;
    using TensorDescriptor  = std::shared_ptr<TensorDescriptorO>;

    // dimension and descriptor
    Dimension        dimension {};
    TensorDescriptor descriptor {};

    public:
    Tensor() : dimension(), SArray<Type>{0}, descriptor {nullptr} {}

    template<typename... Values>
    Tensor(int v1, Values... k)
        : dimension(v1, k...), SArray<Type>(Dimension(v1, k...).size()),
          descriptor(std::make_shared<TensorDescriptorO>(Dimension(v1, k...))) {
        this->mallocCpu();
        this->mallocGpu();
    }

    explicit Tensor(const Dimension& p_dimension)
        : SArray<Type>(p_dimension.size()), dimension(p_dimension),
          descriptor(std::make_shared<TensorDescriptorO>(p_dimension)) {
        this->mallocCpu();
        this->mallocGpu();
    }

    Tensor(Tensor&& other) noexcept
        : SArray<Type>(std::move(other)), dimension(std::move(other.dimension)),
          descriptor(std::move(other.descriptor)) {}

    Tensor(const Tensor& other) noexcept
        : SArray<Type>(other), dimension(other.dimension),
          descriptor(std::make_shared<TensorDescriptorO>(other.dimension)) {}

    Tensor& operator=(Tensor&& other) noexcept {
        SArray<Type>::operator=(std::move(other));
        dimension  = std::move(other.dimension);
        descriptor = std::move(other.descriptor);
        return *this;
    }

    Tensor& operator=(const Tensor& other) noexcept {
        SArray<Type>::operator=(other);
        dimension  = other.dimension;
        descriptor = std::make_shared<TensorDescriptorO>(other.dimension);
        return *this;
    }

    // return the rank which equals the amount of indices required to index this tensor
    [[nodiscard]] int rank() const { return dimension.rank(); }

    // returns a reference to the value stored at the given index
    template<typename... Values>
    [[nodiscard]] Type& get(Values... values) {
        return SArray<Type>::get(dimension.index(values...));
    }

    // returns a copy of the value stored at the given index
    template<typename... Values>
    [[nodiscard]] inline Type get(Values... values) const {
        return SArray<Type>::get(dimension.index(values...));
    }

    // returns a reference to the value stored at the given index
    template<typename... Values>
    [[nodiscard]] inline Type& operator()(Values... values) {
        return get(values...);
    }

    // returns a copy of the value stored at the given index
    template<typename... Values>
    [[nodiscard]] inline Type& operator()(Values... values) const {
        return get(values...);
    }

    using SArray<Type>::operator[];

    friend std::ostream& operator<<(std::ostream& os, const Tensor<Type>& tensor) {

        ArraySizeType N = tensor.rank();

        if (N > 4) {
            os << static_cast<SArray<Type>>(tensor) << "\n";
        } else {

            auto make_index = [&tensor, &N](int h1, int h2, int h3, int h4) {
                if (N == 1)
                    return tensor.dimension.index(h4);
                if (N == 2)
                    return tensor.dimension.index(h3, h4);
                if (N == 3)
                    return tensor.dimension.index(h2, h3, h4);
                if (N == 4)
                    return tensor.dimension.index(h1, h2, h3, h4);
                return ArraySizeType (0);
            };

            int dim_1 = 1;
            int dim_2 = 1;
            int dim_3 = 1;
            int dim_4 = 1;

            if (N >= 1)
                dim_4 = tensor.dimension[N - 1];
            if (N >= 2)
                dim_3 = tensor.dimension[N - 2];
            if (N >= 3)
                dim_2 = tensor.dimension[N - 3];
            if (N >= 4)
                dim_1 = tensor.dimension[N - 4];

            for (int i1 = 0; i1 < dim_1; i1++) {
                // iterate over 3rd dimension first for printing
                for (int i3 = 0; i3 < dim_3; i3++) {
                    // print each cell
                    for (int i2 = 0; i2 < dim_2; i2++) {
                        for (int i4 = 0; i4 < dim_4; i4++) {
                            os << std::fixed << std::setw(12) << std::right << std::setprecision(4)
                               << tensor[make_index(i1, i2, i3, i4)];
                        }
                        os << "\t";
                    }
                    os << "\n";
                }
                os << "\n";
            }
        }

        return os;
    }

    const Dimension&        getDimension() const { return dimension; }
    const TensorDescriptor& getDescriptor() const { return descriptor; }
};

#endif    // AD_TENSOR_H
