#pragma once

#include "AbstractActivation.hpp"


namespace CxxSDNN{

class IzhikevichActivation : public AbstractActivation
{
private:
    double input_scale;
    double output_scale;
    double izh_border;
    double param_a;
    double param_b;
    double param_c;
    double param_d;
    double param_e;
    std::uint32_t dim;
    xt::xarray<double> control;
    xt::xarray<double> state;

public:
    explicit IzhikevichActivation(
        double i_scale, double o_scale,
        double _izh_border, double a, 
        double b, double c,
        double d, double e, std::uint32_t _dim
    );

    explicit IzhikevichActivation(
        double i_scale, double o_scale,
        double _izh_border, double a, 
        double b, double c,
        double d, double e
    );

    explicit IzhikevichActivation(
        double i_scale, double o_scale
    );

    explicit IzhikevichActivation(
        double _izh_border
    );

    explicit IzhikevichActivation(
        std::uint32_t _dim
    );

    explicit IzhikevichActivation();
    

    xt::xarray<double> operator()(xt::xarray<double> input, double step) override;
};

};
