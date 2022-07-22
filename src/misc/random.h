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

#ifndef AD_RANDOM_H
#define AD_RANDOM_H

#include <algorithm>
#include <ctime>
#include <random>

namespace random {

inline std::default_random_engine generator {};

void                              seed(int seed = -1);

template<typename Type>
Type normal(Type mean, Type deviation) {

    if constexpr (std::is_floating_point<Type>()) {
        std::normal_distribution<Type> distribution(mean, deviation);
        return distribution(generator);
    } else {
        std::normal_distribution<float> distribution((static_cast<Type>(mean)),
                                                     static_cast<Type>(deviation));
        return static_cast<Type>(distribution(generator));
    }
}

template<typename Type>
Type uniform(Type min, Type max) {
    if constexpr (std::is_floating_point<Type>()) {
        std::uniform_real_distribution<Type> distribution(min, max);
        return distribution(generator);
    } else if constexpr (std::is_integral<Type>()) {
        if constexpr (std::is_same<Type, bool>::value) {
            return static_cast<Type>(round(uniform<int>(0, 1)));
        } else {
            std::uniform_int_distribution<Type> distribution(min, max);
            return distribution(generator);
        }
    }
}

template<typename Type>
void shuffle(Type* begin, int count){
    for(int i = 0; i < count; i++){
        int idx = uniform(0, count-1);
        std::swap(begin[i], begin[idx]);
    }
}

}    // namespace random

#endif    // AD_RANDOM_H
