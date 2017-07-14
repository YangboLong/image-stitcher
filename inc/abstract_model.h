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
/// @File    abstract_model.h
/// @Brief   Abstract model class with abstract parameters for the RANSAC
///          model fitting algorithm
/// @Author  Yangbo Long <yangbo.long.mav@gmail.com>
/// @Version 0.1.0
/// @Date    2017-05-23

#pragma once

#include <iostream>
#include <stdexcept>
#include <vector>
#include <array>
#include <memory>

/// --------------------------------------------------------------------------
/// @Brief  Each abstract model is made of abstract parameters.
/// --------------------------------------------------------------------------
class AbstrParam {
public:
    virtual ~AbstrParam(void) {}; // to make it polymorphic
};

/// --------------------------------------------------------------------------
/// @Brief  Abstract model type for generic RANSAC model fitting with minimum
///         number of parameters required to define this model.
///
/// @tparam t_num_params, non-type template parameter, number of parameters
/// --------------------------------------------------------------------------
template <int t_num_params>
class AbstractModel {
protected:
    std::array<std::shared_ptr<AbstrParam>, t_num_params> min_model_params_;

public:
    virtual void
    initialize(std::vector<std::shared_ptr<AbstrParam>> inputs1,
               std::vector<std::shared_ptr<AbstrParam>> inputs2) = 0;
    virtual std::tuple<double,
                       std::array<std::vector<std::shared_ptr<AbstrParam>>, 2>,
                       std::array<std::array<double, 3>, 3>> evaluate() = 0;

    virtual std::array<std::shared_ptr<AbstrParam>, t_num_params>
    get_model_params(void) { return min_model_params_; };
};

