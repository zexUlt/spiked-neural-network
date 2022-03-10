#include "SpikeDNNet.hpp"

using CxxSDNN::SpikeDNNet;

SpikeDNNet::SpikeDNNet(
        std::function<nc::NdArray<double>(nc::NdArray<double>, double)> act_func_1,
        std::function<nc::NdArray<double>(nc::NdArray<double>, double)> act_func_2,
        nc::NdArray<double> _mat_W_1,
        nc::NdArray<double> _mat_W_2,
        int dim = 2,
        nc::NdArray<double> _mat_A = 20. * nc::diag(nc::NdArray<double>({-1., -2.})),
        nc::NdArray<double> _mat_P = 1575.9 * nc::diag(nc::NdArray<double>({60., 40.})),
        nc::NdArray<double> _mat_K_1 = .15 * nc::diag(nc::NdArray<double>({10., 1.})),
        nc::NdArray<double> _mat_K_2 = .15 * nc::diag(nc::NdArray<double>({1., 1.}))
    ) :
    afunc_1(act_func_1), afunc_2(act_func_2), mat_A(_mat_A),
    mat_P(_mat_P), mat_K_1(_mat_K_1), mat_K_2(_mat_K_2),
    init_mat_W_1(_mat_W_1)
{

}

nc::NdArray<double> SpikeDNNet::moving_average(nc::NdArray<double> x, int w = 2)
{
    // Здесь должна быть другая свертка
    return nc::filter::convolve1d<double>(); 
}

nc::NdArray<nc::NdArray<double>> SpikeDNNet::smooth(nc::NdArray<nc::NdArray<double>> x, int w = 2)
{
    auto outter_shape = x.shape();
    auto inner_shape  = x[0].shape();

    nc::NdArray<nc::NdArray<double>> new_x(nc::Shape(1, outter_shape.cols - w + 1));
    
    for(size_t i = 0; i < new_x.shape().cols; ++i){
        new_x[1, i] = nc::ones<double>(inner_shape);
    }

    
}