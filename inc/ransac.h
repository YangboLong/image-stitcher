/// Copyright (C)
///
/// This program is free software; you can redistribute it and/or
/// modify it under the terms of the GNU General Public License
/// as published by the Free Software Foundation; either version 2
/// of the License, or (at your option) any later version.
///
/// This program is distributed in the hope that it will be useful,
/// but WITHOUT ANY WARRANTY; without even the implied warranty of
/// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
/// GNU General Public License for more details.
///
/// You should have received a copy of the GNU General Public License
/// along with this program; if not, write to the Free Software
/// Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.
///
/// @File    ransac.h
/// @Brief   Template class implementation of RANSAC algorithm
/// @Author  Yangbo Long <yangbo.long.mav@gmail.com>
/// @Version 0.1.0
/// @Date    2017-05-23

#pragma once

#include <iostream>
#include <cmath>
#include <string>
#include <random>
#include <memory>
#include <algorithm>
#include <vector>
#include <omp.h>

#include "abstract_model.h"

template <class T, int t_num_params>
class RANSAC {
private:
    // all the data
    std::vector<std::shared_ptr<AbstrParam>> data1_, data2_;
    // vector of all sampled models
    std::vector<std::shared_ptr<T>> sampled_models_;
    double best_error_; // model's best error found so far
    std::array<std::vector<std::shared_ptr<AbstrParam>>, 2> best_inliers_;

    // number of iterations before termination
    int max_iters_;
    // mersenne twister high quality RNG that support OpenMP multi-threading
    std::vector<std::mt19937> rand_engines_;

public:
    RANSAC(void) {
        int threads = std::max(1, omp_get_max_threads());
        std::cout << "Maximum usable threads: " << threads << std::endl;
        for(int i = 0; i < threads; ++i) {
            std::random_device SeedDevice;
            rand_engines_.push_back(std::mt19937(SeedDevice()));
        }

        reset();
    };

    virtual ~RANSAC(void) {};

    // clear sampled models, etc. and prepare for next call
    void reset(void) {
        data1_.clear();
        data2_.clear();
        sampled_models_.clear();
        best_error_ = std::numeric_limits<double>::max();
        best_inliers_ = {};
    };

    void initialize(int max_iters = 1000) {
        max_iters_ = max_iters;
    };

    const std::array<std::vector<std::shared_ptr<AbstrParam>>, 2>&
    get_best_inliers(void) { return best_inliers_; };

    bool estimate(std::vector<std::shared_ptr<AbstrParam>> data1,
                  std::vector<std::shared_ptr<AbstrParam>> data2) {
        if (data1.size() <= t_num_params | data2.size() <= t_num_params) {
            std::cout << "RANSAC: Number of data points is too few." << std::endl;
            return false;
        }

        data1_ = data1;
        data2_ = data2;

        int data1_size = data1_.size();
        std::uniform_int_distribution<int> UniDist(0, int(data1_size - 1));

        std::vector<double> err(max_iters_);
        std::vector<std::array<std::vector<std::shared_ptr<AbstrParam>>, 2>>
            inliers_accum(max_iters_);
        sampled_models_.resize(max_iters_);

        // divide loop iterations between a group of spawned threads
        int num_threads = std::max(1, omp_get_max_threads());
        omp_set_dynamic(0); // explicitly disable dynamic teams
        omp_set_num_threads(num_threads);

        // Repeatedly call the FundMatModel's constructor and evaluate functions
        // in the for loop below, to obtain the best model using RANSAC algorithm.
#pragma omp parallel for
        for (int i = 0; i < max_iters_; i++) {
            // select t_num_params random samples
            std::vector<std::shared_ptr<AbstrParam>> rand_samples1(t_num_params);
            std::vector<std::shared_ptr<AbstrParam>> rand_samples2(t_num_params);
            // randomized data
            std::vector<std::shared_ptr<AbstrParam>> rd1 = data1_;
            std::vector<std::shared_ptr<AbstrParam>> rd2 = data2_;

            // shuffle in order to pick randomly ordered elements each time
            std::shuffle(rd1.begin(), rd1.end(), rand_engines_[omp_get_thread_num()]);
            std::copy(rd1.begin(), rd1.begin() + t_num_params, rand_samples1.begin());
            std::shuffle(rd2.begin(), rd2.end(), rand_engines_[omp_get_thread_num()]);
            std::copy(rd2.begin(), rd2.begin() + t_num_params, rand_samples2.begin());

            // call constructor of a derived class of AbstractModel
            std::shared_ptr<T> rand_model = std::make_shared<T>(
                    rand_samples1, rand_samples2, data1_, data2_);
            // call evaluate to check if the current model is the best so far
            std::pair<double, std::array<std::vector<std::shared_ptr<AbstrParam>>, 2>>
                eval_pair = rand_model->evaluate();
            err[i] = eval_pair.first;
            inliers_accum[i] = eval_pair.second;

            // push back into history, could be removed later
            // sampled_models_[i] = rand_model;
        }

        for (int i = 0; i < max_iters_; i++) {
            if (err[i] < best_error_) {
                best_error_ = err[i];
                best_inliers_ = inliers_accum[i];
            }
        }

        return true;
    };
};

