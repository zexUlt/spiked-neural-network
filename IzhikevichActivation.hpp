#pragma once

#include "AbstractActivation.hpp"

namespace CxxSDNN{

class IzhikevichActivation : public AbstractActivation
{
private:
    double izh_border;
    double param_a;
    double param_b;
    double param_c;
    double param_d;
    double param_e;
    nc::uint32 dim;
    nc::NdArray<double> control;
    nc::NdArray<double> state;

public:
    explicit IzhikevichActivation(
        double _izh_border, double a, 
        double b, double c,
        double d, double e, nc::uint32 _dim);

    explicit IzhikevichActivation(
        double _izh_border, double a, 
        double b, double c,
        double d, double e
    );

    explicit IzhikevichActivation(
        double _izh_border
    );

    explicit IzhikevichActivation(
        nc::uint32 _dim
    );

    explicit IzhikevichActivation();
    

    nc::NdArray<double> operator()(nc::NdArray<double> input, double step) override;
};

};
