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

#ifndef AD_FORWARD_H
#define AD_FORWARD_H

#include "../../graph/Graph.h"
#include "State.h"
namespace ppo{

int getAction(SArray<float> &probs, bool deterministic=false){
    if(deterministic){
        int action = 0;
        for(int i = 0; i < probs.size(); i++){
            if(probs[i] > probs[action]){
                action = i;
            }
        }
        return action;
    }else{
        float bias = random::uniform(0.00001f,0.99999f);
        float sum = 0;
        for(int i = 0; i < probs.size(); i++){
            sum += probs[i];
            if(sum > bias){
                return i;
            }
        }
        return (int)random::uniform((int)0, (int)probs.size()-1);
    }
}

SArray<float> evaluatePolicy(SArray<float>& state, Graph& graph){
    // set the batch size to 1
    graph.setBatchSize(1);

    // set the input
    graph.getInput<float>(0)->values.copyFrom(state);

    // upload
    graph.uploadInputs();

    // forward
    graph.forward();

    // get the actions
    graph.getOutput()->values.gpuDownload();
    return static_cast<SArray<float>>(graph.getOutput()->values);
}


SArray<float> evaluateValue(SArray<float>& state, Graph& graph){
    // set the batch size to 1
    graph.setBatchSize(1);

    // set the input
    graph.getInput<float>(0)->values.copyFrom(state);

    // upload
    graph.uploadInputs();

    // forward
    graph.forward();

    // get the actions
    graph.getOutput()->values.gpuDownload();
    return static_cast<SArray<float>>(graph.getOutput()->values);
}

}

#endif    // AD_FORWARD_H
