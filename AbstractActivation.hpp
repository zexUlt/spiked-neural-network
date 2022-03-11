#pragma once

#include "NumCpp.hpp"

namespace CxxSDNN{

class AbstractActivation
{
public:
    virtual nc::NdArray<double> operator()(nc::NdArray<double> input, double step = 0) = 0;
    virtual ~AbstractActivation(){}
};

};
