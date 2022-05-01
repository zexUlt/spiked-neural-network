#pragma once

#include <xtensor/xarray.hpp>

namespace CxxSDNN{

class AbstractActivation
{
public:
    virtual xt::xarray<double> operator()(xt::xarray<double> input, double step = 0) = 0;
    virtual ~AbstractActivation() {};
};

};
