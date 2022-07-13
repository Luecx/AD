#include "graph/Graph.h"
#include "graph/Input.h"
#include "graph/Node.h"
#include "graph/activation/CReLU.h"
#include "graph/activation/ReLU.h"
#include "graph/activation/Sigmoid.h"
#include "graph/activation/Softmax.h"
#include "graph/activation/Swish.h"
#include "graph/elementwise/Exp.h"
#include "graph/elementwise/Log.h"
#include "graph/elementwise/Mul.h"
#include "graph/elementwise/Sub.h"
#include "graph/nn/Linear.h"
#include "graph/reduction/Avg.h"
#include "graph/reduction/Mean.h"
#include "graph/reduction/Select.h"
#include "graph/reduction/Sum.h"
#include "misc/random.h"
#include "operations/affine/affine.h"
#include "operations/affine_bp/affine_bp.h"
#include "operations/operations.h"
#include "operations/select/select.h"
#include "operations/sigmoid/sigmoid.h"
#include "rl/ppo/Episode.h"
#include "rl/ppo/Forward.h"
#include "rl/ppo/Iteration.h"
#include "rl/ppo/State.h"
#include "tensor/Dimension.h"
#include "tensor/Tensor.h"

#include <iostream>


SArray<float> get_state(int field[3][3], int active_player){
    SArray<float> res{18};
    res.mallocCpu();
    int idx = 0;
    for(int i = 0; i < 3; i++){
        for(int j = 0; j < 3; j++){
            if(field[i][j] == active_player){
                res[idx] = 1;
            }
            if(field[i][j] == -active_player){
                res[idx+9] = 1;
            }
            idx++;
        }
    }
    return res;
}

std::tuple<ppo::Episode,ppo::Episode> play_game(Graph& policy_network, Graph&  value_network, bool deterministic=false){
    int field[3][3]{};

    ppo::Episode ep_p1{};
    ppo::Episode ep_p2{};


    for(int i = 0; i < 9; i++){
        int player_id = (i % 2 == 0 ? 1:-1);

        SArray<float> state = get_state(field, player_id);
        SArray<float> probs = ppo::evaluatePolicy(state, policy_network);
        SArray<float> value = ppo::evaluateValue (state, value_network);

        int action = ppo::getAction(probs, deterministic);
        int action_x = action / 3;
        int action_y = action % 3;

        ppo::Episode& current_ep = (i % 2 == 0 ? ep_p1:ep_p2);
        ppo::Episode& opponen_ep = (i % 2 == 0 ? ep_p2:ep_p1);
        current_ep.addState(state, probs, value[0], action);

        // check for illegal actions
        if(field[action_x][action_y] != 0){
            current_ep.setRewardForLastAction(-20);
            break;
        }

        field[action_x][action_y] = player_id;

        // check for mate
        if( field[(action_x+1) % 3][action_y] == player_id &&
            field[(action_x+2) % 3][action_y] == player_id){
            current_ep.setRewardForLastAction(10);
            opponen_ep.setRewardForLastAction(-10);
            break;
        }

        // check for mate
        if( field[action_x][(action_y+1) % 3] == player_id &&
            field[action_x][(action_y+2) % 3] == player_id){
            current_ep.setRewardForLastAction(10);
            opponen_ep.setRewardForLastAction(-10);
            break;
        }

        if( field[0][0] == player_id &&
            field[1][1] == player_id &&
            field[2][2] == player_id){
            current_ep.setRewardForLastAction(10);
            opponen_ep.setRewardForLastAction(-10);
            break;
        }

        if( field[2][0] == player_id &&
            field[1][1] == player_id &&
            field[0][2] == player_id){
            current_ep.setRewardForLastAction(10);
            opponen_ep.setRewardForLastAction(-10);
            break;
        }
    }
    ep_p1.finishEpisode(0.7);
    ep_p2.finishEpisode(0.7);

    return {ep_p1, ep_p2};

}

//ppo::Episode play_game(Graph& policy_network, Graph&  value_network){
//
//    int x_coord = 1;
//    int y_coord = 1;
//
//    bool flag_1 = true;
//    bool flag_2 = true;
//
//#define INDEX(x,y) ((y) * 5 + (x))
//
//    ppo::Episode episode{};
//
//    for(int r = 0; r < 25; r++){
//        SArray<float> state{27};
//        state.mallocCpu();
//        state.clear();
//
//        state(x_coord * 5 + y_coord) = 1;
//        state(25) = flag_1;
//        state(26) = flag_2;
//
//        SArray<float> probs = ppo::evaluatePolicy(state, policy_network);
//        SArray<float> value = ppo::evaluateValue (state, value_network);
//        int action = ppo::getAction(probs, false);
//
//        episode.addState(state, probs, value[0], action);
//
//        if(action == 0){
//            x_coord ++;
//        }if(action == 1){
//            x_coord --;
//        }if(action == 2){
//            y_coord ++;
//        }if(action == 3){
//            y_coord --;
//        }
//
//        if(x_coord < 0 || y_coord < 0 || x_coord > 4 || y_coord > 4){
//            episode.setRewardForLastAction(-5);
//            break;
//        }
//
//        if (x_coord == 0 && y_coord == 2 ||
//            x_coord == 1 && y_coord == 0 ||
//            x_coord == 3 && y_coord == 3 ||
//            x_coord == 2 && y_coord == 1){
//            episode.setRewardForLastAction(-10);
//            break;
//        }
//
//
//        if(x_coord == 0 && y_coord == 4 && !flag_1 && !flag_2){
//            episode.setRewardForLastAction(10);
//            break;
//        }
//
//        if (x_coord == 3 && y_coord == 1 && flag_1 && !flag_2) {
//            episode.setRewardForLastAction(7);
//            flag_1 = false;
//        }
//
//        if (x_coord == 2 && y_coord == 3 && flag_2) {
//            episode.setRewardForLastAction(7);
//            flag_2 = false;
//        }
//    }
//#undef INDEX
//
//    episode.finishEpisode(0.5);
//    return episode;
//}

int main() {

//    random::seed(1);

    Graph graph{};
    auto i1 = graph.addInput<float>(Dimension(18));
    auto l1 = graph.addNode<Linear>(i1, 128);
    auto h1 = graph.addNode<ReLU>(l1);
    auto l2 = graph.addNode<Linear>(h1, 9);
    auto s1 = graph.addNode<Softmax>(l2);

    Graph value_network{};
    auto v_i1 = value_network.addInput<float>(Dimension(18));
    auto v_l1 = value_network.addNode<Linear>(v_i1, 128);
    auto v_h1 = value_network.addNode<ReLU>(v_l1);
    auto v_l2 = value_network.addNode<Linear>(v_h1, 1);

    for(int ep = 0; ep < 2048; ep++){
        std::vector<ppo::Episode> episodes{};
        for(int i = 0; i < 2048; i++){
            auto tuple = play_game(graph, value_network);
            episodes.push_back(std::get<0>(tuple));
            episodes.push_back(std::get<1>(tuple));
        }
        ppo::iteration(episodes, graph, value_network);
    }


    graph.saveParams("policy_network.wgt");
    value_network.saveParams("value_network.wgt");

    auto episodes = play_game(graph, value_network, true);
    std::cout << std::get<0>(episodes) << std::endl;
    std::cout << std::get<1>(episodes) << std::endl;




}
