#pragma once

#include "AbstractActivation.hpp"


namespace CxxSDNN{

class SigmoidActivation : public AbstractActivation
{
private: 
    double param_a;
    double param_b;
    double param_c;
    double param_d;

public:
    explicit SigmoidActivation(double a, double b, double c, double d);
    xt::xarray<double> operator()(xt::xarray<double> input, double step) override;
};

};