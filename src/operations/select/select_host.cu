/**
    CudAD is a CUDA neural network trainer, specific for chess engines.
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

#include <iostream>
#include "select.h"

// clang-format off
void select_host(
    const float* A,
    const int  * index,
          float* B,
    unsigned int m,
    unsigned int n){
    // clang-format on

    for (int idy = 0; idy < m; idy++) {
        B[idy] = (A[INDEX_2D(n,idy, (int)index[idy])]);
    }
}
