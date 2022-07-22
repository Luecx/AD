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
// Created by Luecx on 20.07.2022.
//

#ifndef AD_GAME2048_H
#define AD_GAME2048_H

#include <string>
#include <vector>

namespace rl { namespace env { namespace game2048 {

using Power = int;
using Value = int;

enum Direction : int { NORTH, EAST, SOUTH, WEST };

enum Orientation : int { VERTICAL, HORIZONTAL };

struct Field {
    Power power = 0;

    Value get_value() const;

    Power get_power() const;

    bool  empty() const;

    bool  operator==(const Field& rhs) const;

    bool  operator!=(const Field& rhs) const;

    bool  operator<(const Field& rhs) const;

    bool  operator>(const Field& rhs) const;

    bool  operator<=(const Field& rhs) const;

    bool  operator>=(const Field& rhs) const;

    void  operator<<(Field& other);
};

Value Field::get_value() const { return (1 << power); }

Power Field::get_power() const { return power; }

bool  Field::empty() const { return power == 0; }

bool Field::operator==(const Field& rhs) const { return power == rhs.power; }

bool Field::operator!=(const Field& rhs) const { return !(rhs == *this); }

bool Field::operator<(const Field& rhs) const { return power < rhs.power; }

bool Field::operator>(const Field& rhs) const { return rhs < *this; }

bool Field::operator<=(const Field& rhs) const { return !(rhs < *this); }

bool Field::operator>=(const Field& rhs) const { return !(*this < rhs); }

void Field::operator<<(Field& other) {
    other.power += power;
    power = 0;
}

template<int N>
struct Board {
    Field fields[N][N] {};

    private:
    template<Direction direction, bool reverse = false>
    Field& get_field(int orthogonal, int index) {
        int x, y;

        if constexpr (direction == NORTH || direction == SOUTH) {
            x = orthogonal;
        } else {
            y = orthogonal;
        }

        if constexpr (direction == NORTH && !reverse || direction == SOUTH && reverse) {
            y = N - 1 - index;
        }
        if constexpr (direction == SOUTH && !reverse || direction == NORTH && reverse) {
            y = index;
        }
        if constexpr (direction == WEST && !reverse || direction == EAST && reverse) {
            x = N - 1 - index;
        }
        if constexpr (direction == EAST && !reverse || direction == WEST && reverse) {
            x = index;
        }

        return fields[x][y];
    }

    public:
    template<Direction direction>
    bool can_shift() {

        // iterate over orthogonal axis
        for (int i1 = 0; i1 < N; i1++) {

            // iterate over main axis
            for (int i2 = 0; i2 < N - 1; i2++) {

                Field& f1 = get_field<direction, false>(i1, i2);
                Field& f2 = get_field<direction, false>(i1, i2 + 1);

                if (!f1.empty()) {
                    if (f2.empty() || f2 == f1) {
                        return true;
                    }
                }
            }
        }

        return false;
    }

    bool isGameOver(){
        return can_shift<VERTICAL>() || can_shift<HORIZONTAL>();
    }

    template<Direction direction>
    void shift() {
//        ASSERT(can_shift<direction>());

        // remove empty squares
        for (int i1 = 0; i1 < N; i1++) {
            // iterate over main axis
            for (int i2 = 0; i2 < N - 1; i2++) {
                Field& f1 = get_field<direction, true>(i1, i2);
                if (f1.empty()) {
                    for (int i2_rep = i2 + 1; i2_rep < N; i2_rep++) {
                        Field& f2 = get_field<direction, true>(i1, i2_rep);
                        if (!f2.empty()) {
                            std::swap(f1, f2);
                            break;
                        }
                    }
                }
            }
        }

        // first shift them together and later remove the empty squares
        for (int i1 = 0; i1 < N; i1++) {
            // iterate over main axis
            for (int i2 = 0; i2 < N - 1; i2++) {
                Field& f1 = get_field<direction, true>(i1, i2);
                Field& f2 = get_field<direction, true>(i1, i2 + 1);
                if (f1 == f2) {
                    f1 << f2;
                }
            }
        }

        // remove empty squares
        for (int i1 = 0; i1 < N; i1++) {
            // iterate over main axis
            for (int i2 = 0; i2 < N - 1; i2++) {
                Field& f1 = get_field<direction, true>(i1, i2);
                if (f1.empty()) {
                    for (int i2_rep = i2 + 1; i2_rep < N; i2_rep++) {
                        Field& f2 = get_field<direction, true>(i1, i2_rep);
                        if (!f2.empty()) {
                            std::swap(f1, f2);
                            break;
                        }
                    }
                }
            }
        }
    }

    template<Direction direction>
    void move() {
        shift<direction>();

        std::vector<std::tuple<int, int>> candidates {};
        for (int x = 0; x < N; x++) {
            for (int y = 0; y < N; y++) {
                if (operator()(x, y).empty()) {
                    candidates.emplace_back(x, y);
                }
            }
        }

        auto sq_idx                                                     = rand() % candidates.size();
        auto fi_idx                                                     = rand() % 10 == 0 ? 2 : 1;

        auto new_place                                                  = candidates[sq_idx];
             operator()(std::get<0>(new_place), std::get<1>(new_place)) = Field {fi_idx};
    }

    SArray<float> state(){

    }

    Field& operator()(int x, int y) { return fields[x][y]; }

    Field  operator()(int x, int y) const { return fields[x][y]; }

    public:
    friend std::ostream& operator<<(std::ostream& os, const Board& board) {

        auto sep = [&os]() {
            os << "+";
            for (int x = 0; x < N; x++) {
                os << "---+";
            }
            os << "\n";
        };

        sep();
        for (int y = 0; y < N; y++) {
            os << "|";
            for (int x = 0; x < N; x++) {

                //                os << std::format("{:^3}|", (board(x,y).empty() ?
                //                "":std::to_string(board(x,y).get_power())));

                os << std::setw(3)
                   << (board(x, y).empty() ? "" : std::to_string(board(x, y).get_power())) << "|";
            }
            os << "\n";
            sep();
        }
        return os;
    }
};
} } }    // namespace rl::env::game2048

#endif    // AD_GAME2048_H
