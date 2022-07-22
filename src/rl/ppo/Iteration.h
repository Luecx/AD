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

#ifndef AD_ITERATION_H
#define AD_ITERATION_H

#include "../../graph/Graph.h"
#include "Episode.h"
namespace ppo {

void iteration(std::vector<Episode>& episodes,
               Graph&                policy_network,
               Graph&                value_network,
               Optimiser*            policy_optimizer,
               Optimiser*            value_optimizer,
               float                 epsilon      = 0.2,
               int                   policy_iters = 16,
               int                   critic_iters = 16) {

    // build the loss graph for the policy network
    Graph policy_loss_graph {};

    auto acti     = policy_loss_graph.addInput<int  >  (Dimension(1));
    auto adva     = policy_loss_graph.addInput<float>  (Dimension(1));
    auto prev_out = policy_loss_graph.addInput<float>  (policy_network.getOutput()->getPartialDimension());
    auto sel_curr = policy_loss_graph.addNode<Select>  (policy_network.getOutput(), acti);
    auto sel_prev = policy_loss_graph.addNode<Select>  (prev_out, acti);
    auto ratio    = policy_loss_graph.addNode<Div>     (sel_curr, sel_prev);
    auto clip_rat = policy_loss_graph.addNode<Clip>    (ratio   , 1-epsilon, 1+epsilon);
    auto term_1   = policy_loss_graph.addNode<Mul>     (ratio   , adva);
    auto term_2   = policy_loss_graph.addNode<Mul>     (clip_rat, adva);
    auto single   = policy_loss_graph.addNode<Min>     (term_1  , term_2);
    auto p_ls     = policy_loss_graph.addNode<Avg>     (single);

//    auto  acti = policy_loss_graph.addInput<int>(Dimension(1));
//    auto  adva = policy_loss_graph.addInput<float>(Dimension(1));
//    auto  selc = policy_loss_graph.addNode<Select>(policy_network.getOutput(), acti);
//    auto  logs = policy_loss_graph.addNode<Log>(selc);
//    auto  prod = policy_loss_graph.addNode<Mul>(logs, adva);
//    auto  p_ls = policy_loss_graph.addNode<Avg>(prod);

    // loss graph for value network
    Graph value_loss_graph{};
    auto rewa = value_loss_graph.addInput<float>(Dimension(1));
    auto diff = value_loss_graph.addNode<Sub>(value_network.getOutput(), rewa);
    auto powr = value_loss_graph.addNode<Mul>(diff, diff);
    auto v_ls = value_loss_graph.addNode<Avg>(powr);


    // count the total amount of inputs
    int total = 0;
    for (int i = 0; i < episodes.size(); i++) {
        total += episodes[i].states.size();
    }

    // set the batch size
    policy_loss_graph.setBatchSize(total);
    policy_network.setBatchSize(total);
    value_loss_graph.setBatchSize(total);
    value_network.setBatchSize(total);

    // prepare the data (return + action vector) and fill in the probabilities of the current
    // policy
    int idx = 0;
    for (auto& h : episodes) {
        for (auto& s : h.states) {
            for(int i = 0; i < s.m_p_values.size(); i++){
                prev_out->values(idx, i) = s.m_p_values[i];
            }

            acti->values[idx]   = s.m_action;
            adva->values[idx]   = s.m_advantage;
            rewa->values[idx++] = s.m_return;
        }
    }

    // adjust the return vector
    float mean = adva->values.mean();
    float std  = adva->values.std();
    float max  = adva->values.max();
    for (int i = 0; i < adva->values.size(); i++) {
        adva->values[i] = (adva->values[i] - mean) / std::max(1e-12f, std);
    }

    // generate the input for the policy and value network
    idx = 0;
    for (auto& h : episodes) {
        for (auto& s : h.states) {
            for (int i = 0; i < s.m_state.size(); i++) {
                policy_network.getInput(0)->values(idx, i) = s.m_state(i);
                value_network .getInput(0)->values(idx, i) = s.m_state(i);
            }
            idx++;
        }
    }


    // load inputs to gpu
    policy_network      .uploadInputs();
    policy_loss_graph   .uploadInputs();
    value_network       .uploadInputs();
    value_loss_graph    .uploadInputs();

    for(int i = 0; i < policy_iters; i++){
        // all the forward steps
        policy_network.forward();
        policy_loss_graph.forward();
        
        // clear gradients
        policy_loss_graph.clearGradients();
        policy_network.clearGradients();
        
        // seed output gradients, -1 for policy to maximise, +1 for value to minimise mse
        p_ls->gradients[0] = -1;
        p_ls->gradients.gpuUpload();
        
        // backprop
        policy_loss_graph.backwards();
        policy_network.backwards();
        
        // update policies and values
        policy_optimizer->apply(total);
    }
    
    for(int i = 0; i < critic_iters; i++){
        // all the forward steps
        value_network.forward();
        value_loss_graph.forward();
        
        // clear gradients
        value_loss_graph.clearGradients();
        value_network.clearGradients();
        
        // seed output gradients, -1 for value to maximise, +1 for value to minimise mse
        v_ls->gradients[0] = 1;
        v_ls->gradients.gpuUpload();

//        v_ls->downloadValues();
//        std::cout << "value loss: " << v_ls->values[0] << std::endl;

        // backprop
        value_loss_graph.backwards();
        value_network.backwards();
        
        // update policies and values
        value_optimizer->apply(total);

    }


    // print p_ls
    p_ls->values.gpuDownload();
    v_ls->values.gpuDownload();
    // compute mean and max total return
    float mean_return = 0;
    float max_return = -1000000;
    for(auto& h:episodes){
        mean_return += h.totalReward();
        max_return = std::max(max_return, h.totalReward());
    }
    std::cout << "     p_ls: " << -p_ls->values[0]
              << "     v_ls: " << v_ls->values[0]
              << "     mean return: " << mean_return / episodes.size()
              << "     max return: " << max_return << std::endl;
}

}    // namespace ppo

#endif    // AD_ITERATION_H
