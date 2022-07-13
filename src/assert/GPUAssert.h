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

#ifndef AD_GPUASSERT_H
#define AD_GPUASSERT_H

#include "../misc/config.h"
#include <iostream>

inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort)
            exit(code);
    }
}

inline void gpuAssert(cudnnStatus_t code, const char* file, int line, bool abort = true) {
    if (code != 0) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudnnGetErrorString(code), file, line);
        if (abort)
            exit(code);
    }
}

#define CUDA_ASSERT(ans)                                                                             \
    { gpuAssert((ans), __FILE__, __LINE__); }



#endif    // AD_GPUASSERT_H
