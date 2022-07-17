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

#include "CarPole.h"
void CarPole::step() {
    float total_mass = mass_car + mass_pole;
    float sin_t      = std::sin(theta(0));
    float cos_t      = std::cos(theta(0));

    theta(2)         = (gravity * sin_t
                        + cos_t
                          * ((-force(0) - mass_pole * length_pole * theta(1) * theta(1) * sin_t)
                             / (total_mass)))
                       / (length_pole * (4.0f / 3.0f - (mass_pole * cos_t * cos_t) / total_mass));
    coord(2) =
        (force(0) + mass_pole * length_pole * (theta(1) * theta(1) * sin_t - theta(2) * cos_t))
        / total_mass;

    theta.euler_forward(time_difference);
    coord.euler_forward(time_difference);

    time(0) += time_difference;
}
SArray<float> CarPole::state() {
    SArray<float> res {4};
    res.mallocCpu();
    res(0) = coord(0);
    res(1) = coord(1);
    res(2) = theta(0);
    res(3) = theta(1);
    return res;
}
void CarPole::control(int control) {
    if (control == 0) {
        force(0) = -force_magnitude;
    }
    if (control == 1) {
        force(0) = force_magnitude;
    }
}
