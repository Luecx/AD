/**
    CudAD is a CUDA neural network trainer, specific for chess engines.
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

#ifndef CUDAD_SRC_OPTIMIZER_ADAM_H_
#define CUDAD_SRC_OPTIMIZER_ADAM_H_

#include "../operations/operations.h"
#include "Optimiser.h"
#include <vector>

#include <tuple>

struct Adam : Optimiser {
    private:
    std::vector<Tensor<float>>            first_moments {};
    std::vector<Tensor<float>>            second_moments {};
//    std::vector<std::tuple<float, float>> value_ranges {};

    public:
    double       beta1 = 0.9;
    double       beta2 = 0.999;
    double       eps   = 1e-8;

    virtual void createBuffers() {
        for (Tape<float>* t : tunable_values) {
            // clang-format off
            first_moments .push_back(Tensor<float> {t->values.getDimension()});
            second_moments.push_back(Tensor<float> {t->values.getDimension()});
            first_moments [first_moments.size() - 1].mallocGpu();
            second_moments[first_moments.size() - 1].mallocGpu();
            // clang-format on
//            value_ranges.push_back(
//                std::tuple<float, float> {t->min_allowed_value, t->max_allowed_value});
        }
    }
    virtual void apply(int batch_size) {

        for (int i = 0; i < tunable_values.size(); i++) {
            adam<DEVICE>(tunable_values[i]->values,
                         tunable_values[i]->gradients,
                         first_moments[i],
                         second_moments[i],
                         lr,
                         beta1,
                         beta2,
                         eps);

//            auto range = value_ranges[i];
//            auto min   = std::get<0>(range);
//            auto max   = std::get<1>(range);
//            if (min != std::numeric_limits<float>::min()
//                || max != std::numeric_limits<float>::max()) {
//                clamp<DEVICE>(tunable_values[i]->values, min, max);
//            }
        }
    }
    virtual void newEpoch() {}
    virtual void logOverview() {}
};

#endif    // CUDAD_SRC_OPTIMIZER_ADAM_H_
