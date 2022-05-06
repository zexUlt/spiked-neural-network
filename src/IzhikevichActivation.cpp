#include "IzhikevichActivation.hpp"

#include "xtensor-blas/xlinalg.hpp"
#include "xtensor/xmasked_view.hpp"
#include "debug_header.hpp"

using CxxSDNN::IzhikevichActivation;

IzhikevichActivation::IzhikevichActivation(
        double i_scale, double o_scale,
        double _izh_border, double a, 
        double b, double c,
        double d, double e, std::uint32_t _dim) :
        input_scale{i_scale}, output_scale{o_scale}, izh_border{_izh_border}, 
        param_a{a}, param_b{b}, param_c{c}, param_d{d}, param_e{e}, dim{_dim}
{
    control = xt::eval(xt::ones<double>({dim}) * param_b * param_e);
    state   = xt::eval(xt::ones<double>({dim}) * param_e);
}

IzhikevichActivation::IzhikevichActivation(
        double i_scale, double o_scale,
        double _izh_border, double a, 
        double b, double c,
        double d, double e
    ) : IzhikevichActivation(i_scale, o_scale, _izh_border, a, b, c, d, e, 2)
{

}

IzhikevichActivation::IzhikevichActivation(
    double i_scale, double o_scale
) : IzhikevichActivation(i_scale, o_scale, 30, 2e-2, 0.2, -65, 8, -65, 2u)
{

}

IzhikevichActivation::IzhikevichActivation(
        double _izh_border
    ) : IzhikevichActivation(80., 1/60., _izh_border, 2e-2, 0.2, -65, 8, -65)
{

}

IzhikevichActivation::IzhikevichActivation(
        std::uint32_t _dim
    ) : IzhikevichActivation(80., 1/60., 30, 2e-2, 0.2, -65, 8, -65, _dim)
{

}

IzhikevichActivation::IzhikevichActivation() : 
    IzhikevichActivation(80., 1/60., 30, 2e-2, 0.2, -65, 8, -65, 2u) 
{
    
}

xt::xarray<double> IzhikevichActivation::operator()(xt::xarray<double> input, double step = .01)
{
    auto vec_scale = xt::ones<double>({this->dim}); 

    // auto self_state_dot = xt::linalg::dot(this->state, this->state);

    this->state += step / 2 * ( 
        .04 * this->state * this->state + 5. * this->state + 140. - this->control + input * this->input_scale
    );

    this->state += step / 2 * ( 
        .04 * this->state * this->state + 5. * this->state + 140. - this->control + input * this->input_scale
    );

    this->control += step * (
        this->param_a * (
            this->param_b * this->state - this->control
        )
    );

    auto beyond_border = this->state > this->izh_border;
    auto state_beyond = xt::masked_view(this->state, beyond_border);
    auto control_by_state = xt::masked_view(this->control, beyond_border);

    state_beyond = vec_scale * this->param_c;
    control_by_state = vec_scale * this->param_d;

    return this->state * this->output_scale;
}