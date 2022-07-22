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

#ifndef AD_SUDOKU_H
#define AD_SUDOKU_H

#include <algorithm>
#include <iostream>

namespace rl::env {

#define SUDOKU_INDEX(quadrant, square) (quadrant * 3 + square)
struct Sudoku {

    int values[9][9] {};
    int solution[9][9]{};

    Sudoku(int attempts=5) {
        // create initial board
        fill();

        // copy the solution and keep it
        std::memcpy(solution, values, 81 * sizeof(int));

        // remove numbers
        while(attempts > 0){
            int x = random::uniform(0,8);
            int y = random::uniform(0,8);
            while(values[x][y] == 0){
                x = random::uniform(0,8);
                y = random::uniform(0,8);
            }

            // create a backup
            int backup[9][9];
            std::memcpy(backup, values, 81 * sizeof(int));

            int original = values[x][y];
            values[x][y] = 0;

            // check how many solutions exist
            int solutions = 0;
            solve(solutions);

            if (solutions != 1){
                values[x][y] = original;
                attempts--;
            }
        }
    }

    bool isSet(int x, int y){
        return values[x][y] != 0;
    }

    bool set(int x, int y, int value){
        if(solution[x][y] != value){
            return false;
        }
        values[x][y] = value;
        return true;
    }

    SArray<float> state(){
        SArray<float> data{9 * 9 * 10};
        data.mallocCpu();
        for(int x = 0; x < 9; x++){
            for(int y = 0; y < 9; y++){
                data[x * 90 + y * 10 + values[x][y]] = 1;
            }
        }
        return data;
    }

    private:

    bool fill() {
        // find next empty cell
        bool found_empty_cell = false;

        int  possible_values[9] {1, 2, 3, 4, 5, 6, 7, 8, 9};
        int  x_coord, y_coord;
        for (int x = 0; x < 9; x++) {
            for (int y = 0; y < 9; y++) {
                if (found_empty_cell)
                    break;
                if (values[x][y] == 0) {
                    // save coords
                    x_coord = x;
                    y_coord = y;
                    // break the loop
                    found_empty_cell = true;
                    // shuffle possible numbers to insert
                    random::shuffle(possible_values, 9);
                    // go through each number and check if its possible to use it
                    for (int v : possible_values) {
                        if (!rowContains(y, v) && !colContains(x, v) && !squareContains(x, y, v)) {
                            values[x][y] = v;
                            if (!boardContains(0)) {
                                return true;
                            } else {
                                if (fill()) {
                                    return true;
                                }
                            }
                        }
                    }
                }
            }
        }
        values[x_coord][y_coord] = 0;
        return false;
    }

    bool solve(int& counter) {
        // find next empty cell
        bool found_empty_cell = false;

        int  possible_values[9] {1, 2, 3, 4, 5, 6, 7, 8, 9};
        int  x_coord, y_coord;
        for (int x = 0; x < 9; x++) {
            for (int y = 0; y < 9; y++) {
                if (found_empty_cell)
                    break;
                if (values[x][y] == 0) {
                    // save coords
                    x_coord = x;
                    y_coord = y;
                    // break the loop
                    found_empty_cell = true;
                    // shuffle possible numbers to insert
                    random::shuffle(possible_values, 9);
                    // go through each number and check if its possible to use it
                    for (int v : possible_values) {
                        if (!rowContains(y, v) && !colContains(x, v) && !squareContains(x, y, v)) {
                            values[x][y] = v;
                            if (!boardContains(0)) {
                                counter++;
                                break;
                            } else {
                                if (solve(counter)) {
                                    return true;
                                }
                            }
                        }
                    }
                }
            }
        }
        values[x_coord][y_coord] = 0;
        return false;
    }

    bool rowContains(int row, int value) {
        for (int x = 0; x < 9; x++) {
            if (values[x][row] == value)
                return true;
        }
        return false;
    }

    bool colContains(int col, int value) {
        for (int y = 0; y < 9; y++) {
            if (values[col][y] == value)
                return true;
        }
        return false;
    }

    bool squareContains(int x, int y, int value) {
        int sq_x = (x / 3);
        int sq_y = (y / 3);

        for (int x = sq_x * 3; x < (sq_x + 1) * 3; x++) {
            for (int y = sq_y * 3; y < (sq_y + 1) * 3; y++) {
                if (values[x][y] == value)
                    return true;
            }
        }
        return false;
    }

    bool boardContains(int value) {
        for (int x = 0; x < 9; x++) {
            for (int y = 0; y < 9; y++) {
                if (values[x][y] == value)
                    return true;
            }
        }
        return false;
    }

    public:
    friend std::ostream& operator<<(std::ostream& os, const Sudoku& sudoku) {
        for (int y = 0; y < 9; y++) {
            for (int x = 0; x < 9; x++) {
                if (sudoku.values[x][y])
                    os << sudoku.values[x][y];
                else
                    os << ".";
            }
            os << "\n";
        }

        return os;
    }
};
#undef SUDOKU_INDEX
}    // namespace rl::env
#endif    // AD_SUDOKU_H
