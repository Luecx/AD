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

#include <iomanip>
#include <iostream>
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
    void finishEpisode(float last_value = 0, float discount_factor = 0.99, float smoothing = 0.95) {
        // based off
        // https://colab.research.google.com/github/keras-team/keras-io/blob/master/examples/rl/ipynb/ppo_cartpole.ipynb#scrollTo=evHpNniaqEtS

        // create the rewards and value array and malloc the cpu
        SArray<float> rewards{(ArraySizeType)this->states.size() + 1};
        SArray<float> values {(ArraySizeType)this->states.size() + 1};
        rewards.mallocCpu();
        values .mallocCpu();

        // fill in the values from the states
        for(int i = 0; i < this->states.size(); i++){
            rewards[i] = this->states[i].m_reward;
            values [i] = this->states[i].m_q_value;
        }

        // set the last value to be the last value which resulted in the terminal state
        rewards[rewards.size()-1] = last_value;
        values [values .size()-1] = last_value;

        // define deltas
        auto reward_slice    = rewards.slice(0,-1,1);
        auto value_cur_slice = values .slice(0,-1,1);
        auto value_nex_slice = values .slice(1, 0,1);

        auto deltas          = SArray<float>(reward_slice);
        for (int i = 0; i < deltas.size(); i++) {
            deltas[i] = reward_slice[i] + discount_factor * value_nex_slice[i] - value_cur_slice[i];
        }

        // compute advantages and returns
        for (int i = deltas.size() - 1; i >= 0; i--) {
            if (i == (deltas.size() - 1)) {
                this->states[i].m_advantage = deltas[i];
                this->states[i].m_return    = rewards[i];
            } else {
                this->states[i].m_advantage = deltas [i] + this->states[i+1].m_advantage * discount_factor * smoothing;
                this->states[i].m_return    = rewards[i] + this->states[i+1].m_return    * discount_factor;
            }
        }

        //        for (int i = states.size() - 1; i >= 0; i--) {
//
//            State& s = states[i];
//
//            float  td_error;
//            if (i == states.size() - 1) {
//                if(game_over){
//                    s.m_return = s.m_reward + discount_factor * 0;
//                    td_error = 0;
//                }else{
//                    s.m_return = s.m_reward + discount_factor * s.m_q_value;
//                    td_error = s.m_return;
//                }
//            } else {
//                State& s2  = states[i + 1];
//                s.m_return = s.m_reward + discount_factor * s2.m_return;
//                td_error   = s.m_reward + discount_factor * s2.m_q_value - s.m_q_value;
//            }
//            s.m_advantage = s.m_advantage * smoothing * discount_factor + td_error;
//        }
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
    Episode& operator+=(const Episode& other) {
        states.insert(states.end(),
                      std::make_move_iterator(other.states.begin()),
                      std::make_move_iterator(other.states.end()));
        return *this;
    }

    friend std::ostream& operator<<(std::ostream& os, const Episode& episode) {
        os << std::setw(3) << std::right << "#" << std::setw(35) << std::right << "state"
           << std::setw(10) << std::right << "action" << std::setw(10) << std::right << "reward"
           << std::setw(10) << std::right << "p-value" << std::setw(10) << std::right << "q-value"
           << std::setw(10) << std::right << "advantage" << std::setw(10) << std::right << "return"
           << "\n";

        for (int i = 0; i < episode.states.size(); i++) {
            const State& s = episode.states[i];
            os << std::fixed << std::setprecision(4) << std::right << std::setw(3) << std::right
               << (i) << std::setw(35) << std::right
               << ("1D-Vector (" + std::to_string(s.m_state.size()) + " values)") << std::setw(10)
               << std::right << s.m_action << std::setw(10) << std::right << s.m_reward
               << std::setw(10) << std::right << s.m_p_values[s.m_action] << std::setw(10)
               << std::right << s.m_q_value << std::setw(10) << std::right << s.m_advantage
               << std::setw(10) << std::right << s.m_return << "\n";
        }
        return os;
    }
};
}    // namespace ppo

#endif    // AD_EPISODE_H
