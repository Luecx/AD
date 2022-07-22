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

#ifndef AD_SNAKE_H
#define AD_SNAKE_H

#include "../../array/SArray.h"

#include <deque>
#include <ostream>

namespace rl::env {

enum Controls : int { CTRL_FORWARD = 0, CTRL_LEFT = 1, CTRL_RIGHT = 2 };

enum Directions : int { DIR_UP, DIR_LEFT, DIR_DOWN, DIR_RIGHT };

struct Direction {
    // 0 = up
    // 1 = left
    // 2 = down
    // 3 = right
    Directions direction;

    Direction  operator+(Controls control);

    bool       operator==(const Directions& rhs) const { return direction == rhs; }
    bool       operator!=(const Directions& rhs) const { return direction != rhs; }

    Directions operator()() { return direction; }
};

struct Location {
    int      x, y;

    bool     operator==(const Location& rhs) const;
    bool     operator!=(const Location& rhs) const;

    Location operator+(Direction direction);
};

template<int size = 10, int vision = 5>
class Snake {

    private:
    Direction            m_direction = Direction {DIR_UP};
    Controls             m_control   = CTRL_FORWARD;
    Location             m_apple;
    std::deque<Location> m_snake;

    public:
    Snake() {
        // generate head
        Location head {};
        head.x = random::uniform(0, size);
        head.y = random::uniform(0, size);
        m_snake.push_back(head);

        // generate apple
        m_apple.x = random::uniform(0, size-1);
        m_apple.y = random::uniform(0, size-1);

        while (isOccupied(m_apple)) {
            m_apple.x = random::uniform(0, size-1);
            m_apple.y = random::uniform(0, size-1);
        }
    }

    // do a step in the environment
    bool step() {
        // adjust the direction based on the control
        m_direction = m_direction + m_control;

        // create a new head based on the direction
        Location new_head = m_snake.back() + m_direction;

        // append head to snake
        m_snake.push_back(new_head);

        // check if new head is not on the apple, if so remove the last piece of the snake
        if (new_head != m_apple) {
            m_snake.pop_front();
            return false;
        } else {
            if(!isWon()){
                // place apple to a new location which is empty
                while (isOccupied(m_apple)) {
                    m_apple.x = random::uniform(0, size-1);
                    m_apple.y = random::uniform(0, size-1);
                }
            }
            return true;
        }
    }

    int length(){
        return m_snake.size();
    }

    bool isOccupied(Location loc) const {
        if (loc.x >= size || loc.y >= size || loc.x < 0 || loc.y < 0)
            return true;
        for (const Location& g : m_snake) {
            if (loc == g) {
                return true;
            }
        }
        return false;
    }

    int isSnake(Location loc) const {
        int idx = 0;
        for (const Location& g : m_snake) {
            if (loc == g) {
                return idx;
            }
            idx++;
        }
        return -1;
    }

    bool isApple(Location loc) const { return loc == m_apple; }

    bool isHead(Location loc) const { return loc == m_snake.back(); }

    bool isGameOver() {

        // check snake length
        if(isWon()) return true;

        // temporarily remove head
        Location head = m_snake.back();
        m_snake.pop_back();

        // check if head occupies some square
        bool occupied = isOccupied(head);

        // push it back to the end
        m_snake.push_back(head);

        return occupied;
    }

    bool isWon(){
        return length() >= (size * size);
    }

    int score() const { return m_snake.size(); }

    // get a state vector
    SArray<float> state() {

        bool obstacle[vision * 2 + 1][vision * 2 + 1]{};

        for(int x = m_snake.back().x - vision; x <= m_snake.back().x + vision; x++){
            for(int y = m_snake.back().y - vision; y <= m_snake.back().y + vision; y++){
                obstacle[x - m_snake.back().x + vision][y - m_snake.back().y + vision] = isOccupied(Location{x,y});
            }
        }

        auto rotate = [&obstacle](){
            for(int i0 = 0; i0 < vision; i0++){
                for(int i1 = 0; i1 < vision + 1; i1++){
                    Location loc_1{i0,i1};
                    Location loc_2{i1,vision*2-i0};
                    Location loc_3{vision*2-i0, vision*2-i1};
                    Location loc_4{vision*2-i1, i0};

                    auto temp = obstacle[loc_1.x][loc_1.y];
                    obstacle[loc_1.x][loc_1.y] = obstacle[loc_4.x][loc_4.y];
                    obstacle[loc_4.x][loc_4.y] = obstacle[loc_3.x][loc_3.y];
                    obstacle[loc_3.x][loc_3.y] = obstacle[loc_2.x][loc_2.y];
                    obstacle[loc_2.x][loc_2.y] = temp;
                }
            }
        };

        if(m_direction == DIR_LEFT){
            rotate();
            rotate();
            rotate();
        }
        if(m_direction == DIR_DOWN){
            rotate();
            rotate();
        }
        if(m_direction == DIR_RIGHT){
            rotate();
        }


//        for(int i0 = 0; i0 < vision * 2 + 1; i0++) {
//            for (int i1 = 0; i1 < vision * 2 + 1; i1++) {
//                if(obstacle[i1][i0]){
//                    std::cout << "#";
//                }else{
//                    std::cout << ".";
//                }
//            }
//            std::cout << std::endl;
//        }

        SArray<float> res{(vision * 2 + 1) * (vision * 2 + 1) + 1};
        res.mallocCpu();
        int idx = 0;
        for (int i0 = 0; i0 < vision * 2 + 1; i0++) {
            for (int i1 = 0; i1 < vision * 2 + 1; i1++) {
                if (obstacle[i1][i0]) {
                    res[idx] = 1.0f;
                } else {
                    res[idx] = 0.0f;
                }
                idx ++;
            }
        }


        // create two vectors
        Location vec1{};
        Location vec2{};

        vec1 = vec1 + m_direction;
        vec2 = Location{m_apple.x - m_snake.back().x,m_apple.y - m_snake.back().y};

        float angle = std::atan2(vec1.x * vec2.y - vec1.y * vec2.x, vec1.x * vec2.x + vec1.y * vec2.y);

        res[(vision * 2 + 1) * (vision * 2 + 1)] = angle;
        return res;
    }

    // control = 0 -> stear forward
    // control = 1 -> stear left
    // control = 2 -> stear right
    void                 control(int control) {
        m_control = static_cast<Controls>(control);
    }

    friend std::ostream& operator<<(std::ostream& os, const Snake& snake) {
        for (int y = -1; y <= size; y++) {
            for (int x = -1; x <= size; x++) {
                Location temp {x, y};
                if (snake.isHead(temp)) {
                    os << "S";
                } else if (snake.isOccupied(temp)) {
                    os << "#";
                } else if (snake.isApple(temp)) {
                    os << "*";
                } else {
                    os << " ";
                }
            }
            os << "\n";
        }
        os << "direction: " << snake.m_direction.direction << "\n";
        os << "control  : " << snake.m_control << "\n";
        return os;
    }
};
}    // namespace rl::env

#endif    // AD_SNAKE_H
