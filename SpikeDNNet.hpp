#pragma once

#include "NumCpp.hpp"


class AbstractActivation;

namespace CxxSDNN{

class SpikeDNNet
{
private:
    int mat_dim;
    nc::NdArray<double> mat_A;
    nc::NdArray<double> mat_P;
    nc::NdArray<double> mat_K_1;
    nc::NdArray<double> mat_K_2;
    nc::NdArray<double> mat_W_1;
    nc::NdArray<double> mat_W_2;
    nc::NdArray<double> init_mat_W_1;
    nc::NdArray<double> init_mat_W_2;
    nc::DataCube<double> array_hist_W_1;
    nc::DataCube<double> array_hist_W_2;
    nc::NdArray<double> smoothed_W_1;
    nc::NdArray<double> smoothed_W_2;
    AbstractActivation& afunc_1;
    AbstractActivation& afunc_2;

public:
    explicit SpikeDNNet(
        AbstractActivation& act_func_1,
        AbstractActivation& act_func_2,
        nc::NdArray<double> mat_W_1,
        nc::NdArray<double> mat_W_2,
        int dim = 2,
        nc::NdArray<double> mat_A = 20. * nc::diag(nc::NdArray<double>({-1., -2.})),
        nc::NdArray<double> mat_P = 1575.9 * nc::diag(nc::NdArray<double>({60., 40.})),
        nc::NdArray<double> mat_K_1 = .15 * nc::diag(nc::NdArray<double>({10., 1.})),
        nc::NdArray<double> mat_K_2 = .15 * nc::diag(nc::NdArray<double>({1., 1.}))
    );

    static nc::NdArray<double> moving_average(nc::NdArray<double> x, nc::uint32 w = 2);

    nc::DataCube<double> smooth(nc::DataCube<double> x, nc::uint32 w = 2);

    nc::NdArray<double> fit(
        nc::NdArray<double> vec_x,
        nc::NdArray<double> vec_u,
        double step = .01,
        int n_epochs = 3,
        int k_points = 2);

    nc::NdArray<double> predict(
        nc::NdArray<double> init_state,
        nc::NdArray<double> vec_u,
        double step = .01);
};

};