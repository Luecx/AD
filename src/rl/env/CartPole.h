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
// Created by Luecx on 14.07.2022.
//

#ifndef AD_CARTPOLE_H
#define AD_CARTPOLE_H

#include "../../array/SArray.h"

namespace rl::env {

struct Variable {
    float der_0 = 0;
    float der_1 = 0;
    float der_2 = 0;

    // adjusts variable and first derivative
    void euler_forward(float delta_t) {
        der_0 += der_1 * delta_t;
        der_1 += der_2 * delta_t;
    }

    float& operator()(int idx = 0) {
        if (idx == 0)
            return der_0;
        if (idx == 1)
            return der_1;
        if (idx == 2)
            return der_2;
    }
};

class CartPole {
    Variable theta {random::uniform(-0.05f, 0.05f), random::uniform(-0.05f, 0.05f)};
    Variable coord {random::uniform(-0.05f, 0.05f), random::uniform(-0.05f, 0.05f)};

    // only use value for force / accumulated time
    Variable    force;
    Variable    time;

    const float gravity         = 9.81;    // gravity pulling the pole down
    const float mass_car        = 1.0;      // mass of the car
    const float mass_pole       = 0.1;       // mass of the pole
    const float length_pole     = 0.5;     // length of the pole

    const float time_difference = 0.02;    // time difference for integration
    const float force_magnitude = 10;      // force for input

    public:
    CartPole(bool swingup=false);

    // do a step in the environment
    void step();

    // get a state vector
    SArray<float> state();

    // control = 0 -> stear left
    // control = 1 -> stear right
    void control(int control);
};
}    // namespace rl::env

#endif    // AD_CARTPOLE_H
