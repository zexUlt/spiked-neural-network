#pragma once

#include "NumCpp.hpp"

namespace CxxSDNN{

class AbstractActivation
{
public:
    AbstractActivation() = default;
    virtual nc::NdArray<double> map(nc::NdArray<double> input, double step) = 0;

};

};
