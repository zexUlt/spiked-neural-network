#include "IzhikevichActivation.hpp"

#include <iostream>

using CxxSDNN::IzhikevichActivation;

IzhikevichActivation::IzhikevichActivation(
        double _izh_border = .18, double a = 2e-5, 
        double b = 35e-3, double c = -55e-3,
        double d = .05, double e = -65e-3, double _dim = 2.) :
        izh_border{_izh_border}, param_a{a}, param_b{b}, param_c{c},
        param_d{d}, param_e{e}, dim{dim}
{
    control = nc::ones<double>(dim) * param_b * param_e;
    state   = nc::ones<double>(dim) * param_e;
}

nc::NdArray<double> IzhikevichActivation::operator()(nc::NdArray<double> input, double step = .01)
{
    auto vec_scale = nc::ones<double>(this->dim);
    auto _state = this->state + step * ( 
        .04 * nc::matmul(this->state, this->state) + 
        5. * this->state + 140. - this->control + input
    );
    
    this->control += step * (
        this->param_a * (
            this->param_b * this->state - this->control
        )
    );
    
    std::cout << nc::all(_state > this->izh_border);

    if(nc::all(_state > this->izh_border)[0]){
        this->state = vec_scale * this->param_c;
        this->control = vec_scale * this->param_d;
    }else{
        this->state = _state;
    }

    return this->state;
}