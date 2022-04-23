#include "IzhikevichActivation.hpp"

#include "xtensor-blas/xlinalg.hpp"
#include "debug_header.hpp"

using CxxSDNN::IzhikevichActivation;

IzhikevichActivation::IzhikevichActivation(
        double _izh_border, double a, 
        double b, double c,
        double d, double e, std::uint32_t _dim) :
        izh_border{_izh_border}, param_a{a}, param_b{b}, param_c{c},
        param_d{d}, param_e{e}, dim{_dim}
{
    control = xt::eval(xt::ones<double>({dim}) * param_b * param_e);
    state   = xt::eval(xt::ones<double>({dim}) * param_e);
}

IzhikevichActivation::IzhikevichActivation(
        double _izh_border, double a, 
        double b, double c,
        double d, double e
    ) : IzhikevichActivation(_izh_border, a, b, c, d, e, 2)
{

}

IzhikevichActivation::IzhikevichActivation(
        double _izh_border
    ) : IzhikevichActivation(_izh_border, 2e-5, 35e-3, -55e-3, .05, -65e-3)
{

}

IzhikevichActivation::IzhikevichActivation(
        std::uint32_t _dim
    ) : IzhikevichActivation(.18, 2e-5, 35e-3, -55e-3, .05, -65e-3, _dim)
{

}

IzhikevichActivation::IzhikevichActivation() : 
    IzhikevichActivation(.18, 2e-5, 35e-3, -55e-3, .05, -65e-3, 2u) 
{
    
}

xt::xarray<double> IzhikevichActivation::operator()(xt::xarray<double> input, double step = .01)
{
    auto vec_scale = xt::ones<double>({this->dim}); 

    auto self_state_dot = xt::linalg::dot(this->state, this->state);

    auto _state = this->state + step * ( 
        .04 * self_state_dot + 5. * this->state + 140. - this->control + input
    );
    
    this->control += step * (
        this->param_a * (
            this->param_b * this->state - this->control
        )
    );

    if(xt::all(_state > this->izh_border)){
        this->state = vec_scale * this->param_c;
        this->control = vec_scale * this->param_d;
    }else{
        this->state = _state;
    }

    return this->state;
}