#include "graph/Graph.h"
#include "graph/Input.h"
#include "graph/Node.h"
#include "graph/activation/CReLU.h"
#include "graph/activation/Clip.h"
#include "graph/activation/ReLU.h"
#include "graph/activation/Sigmoid.h"
#include "graph/activation/Softmax.h"
#include "graph/activation/Swish.h"
#include "graph/elementwise/Div.h"
#include "graph/elementwise/Exp.h"
#include "graph/elementwise/Log.h"
#include "graph/elementwise/Mul.h"
#include "graph/elementwise/Sub.h"
#include "graph/nn/Linear.h"
#include "graph/reduction/Avg.h"
#include "graph/reduction/Mean.h"
#include "graph/reduction/Min.h"
#include "graph/reduction/Select.h"
#include "graph/reduction/Sum.h"
#include "misc/random.h"
#include "misc/timer.h"
#include "operations/affine/affine.h"
#include "operations/affine_bp/affine_bp.h"
#include "operations/operations.h"
#include "operations/select/select.h"
#include "operations/sigmoid/sigmoid.h"
#include "optimizer/Adam.h"
#include "rl/env/CartPole.h"
#include "rl/env/Game2048.h"
#include "rl/env/Snake.h"
#include "rl/env/Sudoku.h"
#include "rl/ppo/Episode.h"
#include "rl/ppo/Forward.h"
#include "rl/ppo/Iteration.h"
#include "rl/ppo/State.h"
#include "tensor/Dimension.h"
#include "tensor/Tensor.h"

#include <iostream>

namespace tic_tac_toe {
SArray<float> get_state(int field[3][3], int active_player) {
    SArray<float> res {18};
    res.mallocCpu();
    int idx = 0;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            if (field[i][j] == active_player) {
                res[idx] = 1;
            }
            if (field[i][j] == -active_player) {
                res[idx + 9] = 1;
            }
            idx++;
        }
    }
    return res;
}

std::tuple<ppo::Episode, ppo::Episode>
    playGame(Graph& policy_network, Graph& value_network, bool deterministic = false) {
    int          field[3][3] {};

    ppo::Episode ep_p1 {};
    ppo::Episode ep_p2 {};

    int          i;
    for (i = 0; i < 9; i++) {
        int           player_id  = (i % 2 == 0 ? 1 : -1);

        SArray<float> state      = get_state(field, player_id);
        SArray<float> probs      = ppo::evaluatePolicy(state, policy_network);
        SArray<float> value      = ppo::evaluateValue(state, value_network);

        int           action     = ppo::getAction(probs, deterministic);
        int           action_x   = action / 3;
        int           action_y   = action % 3;

        ppo::Episode& current_ep = (i % 2 == 0 ? ep_p1 : ep_p2);
        ppo::Episode& opponen_ep = (i % 2 == 0 ? ep_p2 : ep_p1);
        current_ep.addState(state, probs, value[0], action);

        // check for illegal actions
        if (field[action_x][action_y] != 0) {
            current_ep.setRewardForLastAction(-20);
            break;
        }

        field[action_x][action_y] = player_id;

        // check for mate
        if (field[(action_x + 1) % 3][action_y] == player_id
            && field[(action_x + 2) % 3][action_y] == player_id) {
            current_ep.setRewardForLastAction(10);
            opponen_ep.setRewardForLastAction(-10);
            break;
        }

        // check for mate
        if (field[action_x][(action_y + 1) % 3] == player_id
            && field[action_x][(action_y + 2) % 3] == player_id) {
            current_ep.setRewardForLastAction(10);
            opponen_ep.setRewardForLastAction(-10);
            break;
        }

        if (field[0][0] == player_id && field[1][1] == player_id && field[2][2] == player_id) {
            current_ep.setRewardForLastAction(10);
            opponen_ep.setRewardForLastAction(-10);
            break;
        }

        if (field[2][0] == player_id && field[1][1] == player_id && field[0][2] == player_id) {
            current_ep.setRewardForLastAction(10);
            opponen_ep.setRewardForLastAction(-10);
            break;
        }
    }

    //     if(i > 8){
    //         ep_p1.setRewardForLastAction(0);
    //         ep_p2.setRewardForLastAction(0);
    //     }

    ep_p1.finishEpisode(0, 0.95, 0.95);
    ep_p2.finishEpisode(0, 0.95, 0.95);

    return {ep_p1, ep_p2};
}
}    // namespace tic_tac_toe

namespace flag_race {
ppo::Episode playGame(Graph& policy_network, Graph& value_network, bool deterministic) {

    int  x_coord = 1;
    int  y_coord = 1;

    bool flag_1  = true;
    bool flag_2  = true;

#define INDEX(x, y) ((y) *5 + (x))

    ppo::Episode episode {};

    for (int r = 0; r < 25; r++) {

        SArray<float> state {27};
        state.mallocCpu();
        state.clear();

        state(x_coord * 5 + y_coord) = 1;
        state(25)                    = flag_1;
        state(26)                    = flag_2;

        SArray<float> probs          = ppo::evaluatePolicy(state, policy_network);
        SArray<float> value          = ppo::evaluateValue(state, value_network);
        int           action         = ppo::getAction(probs, deterministic);

        episode.addState(state, probs, value[0], action);

        if (action == 0) {
            x_coord++;
        }
        if (action == 1) {
            x_coord--;
        }
        if (action == 2) {
            y_coord++;
        }
        if (action == 3) {
            y_coord--;
        }

        if (x_coord < 0 || y_coord < 0 || x_coord > 4 || y_coord > 4) {
            episode.setRewardForLastAction(-5);
            break;
        }

        if (x_coord == 0 && y_coord == 2 || x_coord == 1 && y_coord == 0
            || x_coord == 3 && y_coord == 3 || x_coord == 2 && y_coord == 1) {
            episode.setRewardForLastAction(-10);
            break;
        }

        if (x_coord == 0 && y_coord == 4 && !flag_1 && !flag_2) {
            episode.setRewardForLastAction(10);
            break;
        }

        if (x_coord == 3 && y_coord == 1 && flag_1 && !flag_2) {
            episode.setRewardForLastAction(7);
            flag_1 = false;
        }

        if (x_coord == 2 && y_coord == 3 && flag_2) {
            episode.setRewardForLastAction(7);
            flag_2 = false;
        }
    }
#undef INDEX

    episode.finishEpisode(0, 0.95, 0.95);
    return episode;
}
}    // namespace flag_race

namespace cartpole {

ppo::Episode playGame(Graph& policy_net, Graph& value_net, bool deterministic = false) {
    rl::env::CartPole pole {};

    ppo::Episode      res {};

    bool              game_over = false;

    for (int i = 0; i < 3000; i++) {
        auto          state  = pole.state();

        SArray<float> probs  = ppo::evaluatePolicy(state, policy_net);
        SArray<float> value  = ppo::evaluateValue(state, value_net);
        int           action = ppo::getAction(probs, deterministic);


        res.addState(state, probs, value[0], action);

        pole.control(action);
        pole.step();

        if (abs(state(0)) > 2.4 || abs(state(2)) > 0.2) {
            game_over = true;
            break;
        }

        res.setRewardForLastAction(1 - abs(state(0)) / 10.0f);
    }

    float last_value = 0;

    if (!game_over) {
        auto          state = pole.state();
        SArray<float> value = ppo::evaluateValue(state, value_net);
        last_value          = value[0];
    }
    res.finishEpisode(last_value, 0.99, 0.95);
    return res;
}
}    // namespace cartpole

namespace cartpole_swingup {

ppo::Episode playGame(Graph& policy_net, Graph& value_net, bool deterministic = false) {
    rl::env::CartPole pole {true};

    ppo::Episode      res {};

    bool              game_over = false;

    for (int i = 0; i < 1000; i++) {
        auto          state  = pole.state();

        SArray<float> probs  = ppo::evaluatePolicy(state, policy_net);
        SArray<float> value  = ppo::evaluateValue(state, value_net);
        int           action = ppo::getAction(probs, deterministic);


        res.addState(state, probs, value[0], action);

        pole.control(action);
        pole.step();

        constexpr float PI =  3.1415926535;

        if (abs(state(0)) > 2.4) {
            res.setRewardForLastAction(-10000);
            game_over = true;
            break;
        }

        if(abs(state(2)) > 5 * PI){
            res.setRewardForLastAction(-10000);
            game_over = true;
            break;
        }

        // bring angle into -pi to pi range
        float angle = state(2);
        while(angle >PI){
            angle -= 2 * PI;
        }while(angle < -PI){
            angle += 2 * PI;
        }

        res.setRewardForLastAction(std::max(0.0f, 50 * cos(state(2))));
    }

    float last_value = 0;

    if (!game_over) {
        auto          state = pole.state();
        SArray<float> value = ppo::evaluateValue(state, value_net);
        last_value          = value[0];
    }
    res.finishEpisode(last_value, 0.99, 0.95);
    return res;
}
}    // namespace cartpole

namespace snake {

ppo::Episode playGame(Graph& policy_net, Graph& value_net, bool deterministic = false) {
    rl::env::Snake<10> snake {};

    ppo::Episode          res {};

    bool                  game_over = false;

    for (int i = 0; i < 6000; i++) {
        auto          state  = snake.state();

        SArray<float> probs  = ppo::evaluatePolicy(state, policy_net);
        SArray<float> value  = ppo::evaluateValue(state, value_net);
        int           action = ppo::getAction(probs, deterministic);

        res.addState(state, probs, value[0], action);

        snake.control(action);
        if (snake.step()) {
            res.setRewardForLastAction(1);
        }

//        for(int x = 0; x < 10;x ++){
//            for(int y = 0; y < 10; y++){
//                if(snake.isHead(rl::env::Location{x,y})){
//                    std::cout << "2.0 ";
//                }else if(snake.isApple(rl::env::Location{x,y})){
//                    std::cout << "1.0 ";
//                }else if(snake.isOccupied(rl::env::Location{x,y})){
//                    std::cout << (3 + (float)snake.isSnake(rl::env::Location{x,y}) / snake.length()) << " ";
//                }else{
//                    std::cout << "0.0 ";
//                }
//            }
//        }
//        std::cout << "\n";

        if (snake.isGameOver()) {
            game_over = true;
            if(!snake.isWon())
                res.setRewardForLastAction(-10);
            break;
        }
    }

    float last_value = 0;

    if (!game_over) {
        auto          state = snake.state();
        SArray<float> value = ppo::evaluateValue(state, value_net);
        last_value          = value[0];
    }
    res.finishEpisode(last_value, 0.99, 0.95);


    return res;
}

}    // namespace snake

namespace sudoku{

ppo::Episode playGame(Graph& policy_net, Graph& value_net, bool deterministic = false) {
    rl::env::Sudoku sudoku {1};

    ppo::Episode          res {};

    bool                  game_over = false;

    for (int i = 0; i < 1000; i++) {
        auto          state  = sudoku.state();

        SArray<float> probs  = ppo::evaluatePolicy(state, policy_net);
        SArray<float> value  = ppo::evaluateValue(state, value_net);
        int           action = ppo::getAction(probs, deterministic);

        res.addState(state, probs, value[0], action);

        int x_coord = action / 81;
        int y_coord = action / 9;
        int number  = action % 9;

        // dont place somewhere where a number already is set
        if(sudoku.isSet(x_coord, y_coord)){
            game_over = true;
            res.setRewardForLastAction(-5);
            break;
        }

        // place and check if the number was wrong
        if(!sudoku.set(x_coord, y_coord, number)){
            game_over = true;
            res.setRewardForLastAction(-3);
            break;
        }else{
            res.setRewardForLastAction(10);
        }
    }

    float last_value = 0;

    if (!game_over) {
        auto          state = sudoku.state();
        SArray<float> value = ppo::evaluateValue(state, value_net);
        last_value          = value[0];
    }
    res.finishEpisode(last_value, 0.1, 0.95);


    return res;
}
}

//namespace game2048{
//ppo::Episode playGame(Graph& policy_net, Graph& value_net, bool deterministic = false) {
//    rl::env::game2048::Board<4> board {};
//
//    ppo::Episode          res {};
//
//    bool                  game_over = false;
//
//    for (int i = 0; i < 1000; i++) {
//        auto          state  = board.state();
//
//        SArray<float> probs  = ppo::evaluatePolicy(state, policy_net);
//        SArray<float> value  = ppo::evaluateValue(state, value_net);
//        int           action = ppo::getAction(probs, deterministic);
//
//        res.addState(state, probs, value[0], action);
//
//        // place and check if the number was wrong
//        if(!sudoku.set(x_coord, y_coord, number)){
//            game_over = true;
//            res.setRewardForLastAction(-3);
//            break;
//        }else{
//            res.setRewardForLastAction(10);
//        }
//    }
//
//    float last_value = 0;
//
//    if (!game_over) {
//        auto          state = sudoku.state();
//        SArray<float> value = ppo::evaluateValue(state, value_net);
//        last_value          = value[0];
//    }
//    res.finishEpisode(last_value, 0.1, 0.95);
//
//
//    return res;
//}
//}

int main(int argc, char* argv[]) {

    using namespace rl::env;

//    random::seed(118);


    using namespace snake;

    Graph graph {};
    auto  i1 = graph.addInput<float>(Dimension(122));
    auto  l1 = graph.addNode<Linear>(i1, 128);
    auto  a1 = graph.addNode<Sigmoid>(l1);
    auto  l3 = graph.addNode<Linear>(a1, 4);
    auto  s1 = graph.addNode<Softmax>(l3);
    l3->m_weights.values.randomiseGaussian(0, 0.01f);
    l3->m_weights.values.gpuUpload();

    Graph value_network {};
    auto  v_i1 = value_network.addInput<float>(Dimension(122));
    auto  v_l1 = value_network.addNode<Linear>(v_i1, 128);
    auto  v_a1 = value_network.addNode<Sigmoid>(v_l1);
    auto  v_l2 = value_network.addNode<Linear>(v_a1, 1);
    v_l2->m_weights.values.randomiseGaussian(0, 0.01f);
    v_l2->m_weights.values.gpuUpload();

    Adam adam1 {};
    Adam adam2 {};
    adam1.init(graph.params());
    adam2.init(value_network.params());

    adam1.lr = 0.001;
    adam2.lr = 0.001;

//    graph.loadParams("snake_policy5.wgt");
//    value_network.loadParams("snake_value5.wgt");
//    for (int ep = 0; ep < 2048; ep++) {
//        std::vector<ppo::Episode> episodes {};
//        for (int i = 0; i < 256; i++) {
//            std::cout << "\repisode " << i << std::flush;
//            auto res = playGame(graph, value_network);
//            episodes.push_back(res);
//        }
//        std::cout << "\r";
//        ppo::iteration(episodes, graph, value_network, &adam1, &adam2, 0.2, 24, 24);
//
//        graph.saveParams("snake_policy5.wgt");
//        value_network.saveParams("snake_value5.wgt");
//    }

    graph.loadParams("../resources/other/networks/rl/snake/snake_policy5.wgt");
    value_network.loadParams("../resources/other/networks/rl/snake/snake_value5.wgt");
    for(int s = 0; s < 20; s++){
        random::seed(s);
        auto episodes = playGame(graph, value_network, false);
        std::cout << "   reward: " << episodes.totalReward() << " " << episodes.states.size() << std::endl;
    }




//    for(auto& s:episodes.states){
//        std::cout << s.m_state << std::endl;
//    }
}