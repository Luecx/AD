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

#ifndef AD_DIMENSION_H
#define AD_DIMENSION_H

#include "../array/Array.h"

#include <cstdint>
#include <ostream>
#include <iomanip>
#include <vector>

using DimensionType = int8_t;

#define INDEX_2D(size_n, m, n) m * size_n + n

struct Dimension {
    private:
    std::vector<ArraySizeType> m_dimension;
    std::vector<ArraySizeType> m_accumulated_size;

    DimensionType              m_rank;

    public:

    Dimension(){

    }

    template<typename... Values>
    Dimension(Values... values) {
        DimensionType dim = 0;
        for (const ArraySizeType p : {values...}) {
            m_dimension.push_back(p);
            dim++;
        }
        m_rank = dim;
        recompute_accumulated();
    }

    explicit Dimension(std::initializer_list<ArraySizeType> dimension)
        : m_dimension(dimension.begin(), dimension.end()), m_rank(dimension.size()) {
        recompute_accumulated();
    }

    private:
    void recompute_accumulated() {
        m_accumulated_size.resize(m_rank);
        if(m_rank == 0) return;
        m_accumulated_size[m_rank - 1] = 1;
        for (DimensionType i = m_rank - 2; i >= 0; i--) {
            m_accumulated_size[i] = m_accumulated_size[i + 1] * m_dimension[i + 1];
        }
    }

    public:
    inline ArraySizeType size() const { return m_dimension[0] * m_accumulated_size[0]; }

    inline ArraySizeType rank() const { return m_rank; }

    inline ArraySizeType accumulated(DimensionType dimension) const {
        return m_accumulated_size[dimension];
    }

    inline ArraySizeType operator[](DimensionType dimension) const { return m_dimension[dimension]; }

    inline ArraySizeType operator()(DimensionType dimension) const { return m_dimension[dimension]; }

    template<typename... Values>
    inline ArraySizeType index(Values... values) const {
        ArraySizeType res = 0;
        DimensionType dim = 0;
        for (const auto p : {values...}) {
            res += p * m_accumulated_size[dim++];
        }
        return res;
    }

    Dimension operator+(ArraySizeType dim) const {
        Dimension d {*this};
        d += dim;
        return d;
    }

    Dimension operator+(const Dimension& dimension) const {
        Dimension d {*this};
        d += dimension;
        return d;
    }

    Dimension& operator+=(ArraySizeType dim) {
        m_rank++;
        m_dimension.push_back(dim);
        recompute_accumulated();
        return *this;
    }

    Dimension& operator+=(const Dimension& dimension) {
        for (DimensionType i = 0; i < dimension.m_rank; i++) {
            this->operator+=(dimension(i));
        }
        recompute_accumulated();
        return *this;
    }

    bool operator==(const Dimension& rhs) const {

        if (m_rank != rhs.rank())
            return false;

        for (int i = 0; i < m_rank; i++) {
            if (m_dimension[i] != rhs(i))
                return false;
        }
        return true;
    }
    bool operator!=(const Dimension& rhs) const { return !(rhs == *this); }

    friend std::ostream& operator<<(std::ostream& os, const Dimension& dimension) {
        for(int i = 0; i < dimension.rank(); i++){
            os << std::fixed << std::setw(12) << std::right << std::setprecision(4)
               << dimension[i];
        }
        return os;
    }
};

#endif    // AD_DIMENSION_H
