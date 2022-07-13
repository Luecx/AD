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
// Created by Luecx on 09.07.2022.
//

#ifndef AD_INPUT_H
#define AD_INPUT_H

#include "Node.h"
template<typename Type>
struct Input : Node<Type>{
    Input(Dimension d) : Node<Type>{d}{}

    private:
    void      clearWeightGradients() override {}
    public:
    void      forward() override {}
    void      backwards() override {}
};

#endif    // AD_INPUT_H
