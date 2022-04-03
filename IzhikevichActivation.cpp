#include "IzhikevichActivation.hpp"


using CxxSDNN::IzhikevichActivation;

IzhikevichActivation::IzhikevichActivation(
        double _izh_border, double a, 
        double b, double c,
        double d, double e, nc::uint32 _dim) :
        izh_border{_izh_border}, param_a{a}, param_b{b}, param_c{c},
        param_d{d}, param_e{e}, dim{_dim}
{
    control = nc::ones<double>(1, dim) * param_b * param_e;
    state   = nc::ones<double>(1, dim) * param_e;
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
        nc::uint32 _dim
    ) : IzhikevichActivation(.18, 2e-5, 35e-3, -55e-3, .05, -65e-3, _dim)
{

}

IzhikevichActivation::IzhikevichActivation() : 
    IzhikevichActivation(.18, 2e-5, 35e-3, -55e-3, .05, -65e-3, 2u) 
{
    
}

nc::NdArray<double> IzhikevichActivation::operator()(nc::NdArray<double> input, double step = .01)
{
    auto vec_scale = nc::ones<double>(1, this->dim);
    auto self_state_norm = nc::matmul(this->state, this->state)[0]; 
    auto _state = this->state + step * ( 
        .04 * self_state_norm + 5. * this->state + 140. - this->control + input
    );
    
    this->control += step * (
        this->param_a * (
            this->param_b * this->state - this->control
        )
    );

    if(nc::all(_state > this->izh_border)[0]){
        this->state = vec_scale * this->param_c;
        this->control = vec_scale * this->param_d;
    }else{
        this->state = _state;
    }

    return this->state;
}