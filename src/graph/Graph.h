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

#ifndef AD_GRAPH_H
#define AD_GRAPH_H

#include "Input.h"
#include "Node.h"

struct Graph {

    using NodePtr = std::unique_ptr<NodeInterface>;

    std::vector<NodePtr> nodes {};
    std::vector<NodePtr> inputs[NIT_N_TYPES] {};

    private:

    int batch_size = -1;
    public:

    template<typename T, typename... Args>
    T* addNode(Args... args) {
        nodes.emplace_back(std::make_unique<T>(args...));
        NodeInterface* raw_ptr = nodes[nodes.size() - 1].get();
        return static_cast<T*>(raw_ptr);
    }

    Node<float>* getOutput() {
        ERROR(!nodes.empty(), "no nodes to compute present in the graph");
        ERROR(nodes.back().get()->getNodeType() == NIT_FLOAT,
              "graph does not allow for non-floating operations to be last");
        NodePtr&       node_ptr = nodes.back();
        NodeInterface* raw_ptr  = node_ptr.get();
        return static_cast<Node<float>*>(raw_ptr);
    }

    template<typename T = float, typename... Args>
    Input<T>* addInput(Args... args) {
        inputs[deriveNodeType<T>()].emplace_back(std::make_unique<Input<T>>(args...));
        return getInput<T>(inputs[deriveNodeType<T>()].size() - 1);
    }

    template<typename T = float>
    Input<T>* getInput(int idx) {
        NodePtr&       node_ptr = inputs[deriveNodeType<T>()].at(idx);
        NodeInterface* raw_ptr  = node_ptr.get();
        return static_cast<Input<T>*>(raw_ptr);
    }

    void uploadInputs() {
        for (auto& h : inputs) {
            for (auto& k : h) {
                k.get()->uploadValues();
            }
        }
    }

    void setBatchSize(int p_batch_size) {

        if(p_batch_size == batch_size){
            return;
        }

        batch_size = p_batch_size;
        for (auto& h : inputs) {
            for (auto& k : h) {
                k->setBatchSize(batch_size);
            }
        }
        for (auto& k : nodes) {
            k->setBatchSize(batch_size);
        }
    }

    void clearGradients(){
        for (auto& h : inputs) {
            for (auto& k : h) {
                k->clearGradients();
            }
        }
        for (auto& k : nodes) {
            k->clearGradients();
        }
    }

    void forward() {
        ERROR(batch_size > 0, "the batch size has not been set or is mis-configured");

        for (auto& k : nodes) {
            k->forward();
        }
    }

    void backwards() {
        for(int i = nodes.size()-1; i>=0; i--){
            nodes[i]->backwards();
        }
    }

    std::vector<Tape<float>*> params(){
        std::vector<Tape<float>*> res{};
        for (auto& k : nodes){
            auto k_params = k->params();
            for(auto& h:k_params){
                res.push_back(h);
            }
        }
        return res;
    }

    void saveParams(const std::string& file){
        FILE* f = fopen(file.c_str(), "wb");

        // figure out how many entries we will store
        uint64_t count = 0;
        for (auto& k : nodes){
            auto k_params = k->params();
            for(auto& h:k_params){
                count += h->values.size();
            }
        }

        fwrite(&count, sizeof(uint64_t), 1, f);

        for (auto& k : nodes){
            auto k_params = k->params();
            for(auto& h:k_params){
                h->values.gpuDownload();
                fwrite(h->values.address<HOST>(), sizeof(float), h->values.size(), f);
            }
        }

        fclose(f);
    }

    void loadParams(const std::string& file){
        FILE* f = fopen(file.c_str(), "rb");

        // figure out how many entries we will store
        uint64_t count = 0;
        for (auto& k : nodes){
            auto k_params = k->params();
            for(auto& h:k_params){
                count += h->values.size();
            }
        }

        uint64_t fileCount = 0;
        fread(&fileCount, sizeof(uint64_t), 1, f);
        ERROR(count == fileCount, "cannot load this file since the amount of parameters stored does not match the graph");

        for (auto& k : nodes){
            auto k_params = k->params();
            for(auto& h:k_params){
                fread(h->values.address<HOST>(), sizeof(float), h->values.size(), f);
                h->values.gpuUpload();
            }
        }
        fclose(f);
    }

    Graph() {}
    Graph(Graph&& other)                 = delete;
    Graph(const Graph& other)            = delete;
    Graph& operator=(Graph&& other)      = delete;
    Graph& operator=(const Graph& other) = delete;

};

#endif    // AD_GRAPH_H
