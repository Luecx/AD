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
// Created by Luecx on 19.07.2022.
//

#include "Snake.h"
bool rl::env::Location::operator==(const rl::env::Location& rhs) const {
    return x == rhs.x && y == rhs.y;
}
bool rl::env::Location::operator!=(const rl::env::Location& rhs) const { return !(rhs == *this); }
rl::env::Location rl::env::Location::operator+(rl::env::Direction direction) {
    Location location {x, y};
    if (direction == DIR_UP) {
        location.y--;
    }
    if (direction == DIR_LEFT) {
        location.x--;
    }
    if (direction == DIR_DOWN) {
        location.y++;
    }
    if (direction == DIR_RIGHT) {
        location.x++;
    }
    return location;
}

rl::env::Direction rl::env::Direction::operator+(rl::env::Controls control) {
    if (control == CTRL_FORWARD)
        return Direction {direction};
    if (control == CTRL_LEFT)
        return Direction {static_cast<Directions>((direction + 1) % 4)};
    if (control == CTRL_RIGHT)
        return Direction {static_cast<Directions>((direction + 3) % 4)};
    return Direction {direction};
}
