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
// Created by Luecx on 10.07.2022.
//

#ifndef AD_STATE_H
#define AD_STATE_H

#include "../../array/SArray.h"

#include <vector>

namespace ppo {

struct State {
    // the state
    SArray<float> m_state;
    // the p-value(s) coming from the policy network
    SArray<float> m_p_values;
    // the quality value coming from the value network
    float m_q_value;
    // the reward from taking the action in the given state
    float m_reward;
    // the action taken
    int m_action;

    // advantage based on q value, reward and future q-values
    float m_advantage;
    // return based on advantage and value function
    float m_return;
};


}    // namespace ppo
#endif    // AD_STATE_H
