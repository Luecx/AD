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
// Created by Luecx on 12.07.2022.
//

#ifndef AD_EPISODE_H
#define AD_EPISODE_H

#include "State.h"

#include <iostream>
#include <iomanip>
#include <ostream>
#include <string>

namespace ppo {

struct Episode {
    std::vector<State> states {};

    // add a state to the states list
    void addState(const SArray<float>& state,
                  const SArray<float>& p_values,
                  float                q_value,
                  int                  action,
                  float                reward = 0) {
        states.push_back({state, p_values, q_value, reward, action});
    }

    // sets the reward to the previously set state
    void setRewardForLastAction(float reward) { states.back().m_reward = reward; }

    // computes the advantages and returns for this episode
    void finishEpisode(float discount_factor = 0.99, float smoothing = 0.95) {
        for (int i = states.size() - 1; i >= 0; i--) {
            // define delta

            float delta = states[i].m_reward;
            if (i != states.size() - 1) {
                delta += discount_factor * states[i + 1].m_q_value - states[i].m_q_value;
            }
            // update advantage
            states[i].m_advantage = delta;
            if (i != states.size() - 1) {
                states[i].m_advantage += discount_factor * smoothing * states[i + 1].m_advantage;
            }
            // calculate returns
            states[i].m_return = states[i].m_advantage + states[i].m_q_value;
        }
    }

    // computes the total reward
    float totalReward() {
        float sum = 0;
        for (int i = 0; i < states.size(); i++) {
            sum += states[i].m_reward;
        }
        return sum;
    }

    // adds another episode to this episode
    Episode& operator+=(const Episode& other){
        states.insert(states.end(), std::make_move_iterator(other.states.begin()),
                                    std::make_move_iterator(other.states.end()));
        return *this;
    }

    friend std::ostream& operator<<(std::ostream& os, const Episode& episode) {
        os << std::setw(3 ) << std::right << "#"
           << std::setw(35) << std::right << "state"
           << std::setw(10) << std::right << "action"
           << std::setw(10) << std::right << "reward"
           << std::setw(10) << std::right << "p-value"
           << std::setw(10) << std::right << "q-value"
           << std::setw(10) << std::right << "advantage"
           << std::setw(10) << std::right << "return" << "\n";

        for(int i = 0; i < episode.states.size(); i++){
            const State& s = episode.states[i];
            os << std::setw(3 ) << std::right << (i)
               << std::setw(35) << std::right << ("1D-Vector (" + std::to_string(s.m_state.size()) + " values)")
               << std::setw(10) << std::right << s.m_action
               << std::setw(10) << std::right << s.m_reward
               << std::setw(10) << std::right << s.m_p_values[s.m_action]
               << std::setw(10) << std::right << s.m_q_value
               << std::setw(10) << std::right << s.m_advantage
               << std::setw(10) << std::right << s.m_return << "\n";
        }
        return os;
    }
};
}    // namespace ppo

#endif    // AD_EPISODE_H
