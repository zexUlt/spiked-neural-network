#include "SigmoidActivation.hpp"

using CxxSDNN::SigmoidActivation;

SigmoidActivation::SigmoidActivation(double a, double b, double c, double d) :
    param_a{a}, param_b{b}, param_c{c}, param_d{d}
{

}

nc::NdArray<double> SigmoidActivation::operator()(nc::NdArray<double> input, double step = -1.)
{
    return this->param_a / (
        this->param_b + this->param_c * nc::exp(this->param_d * input)
        );
}