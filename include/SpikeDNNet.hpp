#pragma once

#include "xtensor/xarray.hpp"

#include <memory>

namespace CxxSDNN{

class AbstractActivation;

class SpikeDNNet
{
private:
    size_t mat_dim;
    xt::xarray<double> mat_A;
    xt::xarray<double> mat_P;
    xt::xarray<double> mat_K_1;
    xt::xarray<double> mat_K_2;
    xt::xarray<double> mat_W_1;
    xt::xarray<double> mat_W_2;
    xt::xarray<double> init_mat_W_1;
    xt::xarray<double> init_mat_W_2;
    xt::xarray<double> array_hist_W_1;
    xt::xarray<double> array_hist_W_2;
    xt::xarray<double> smoothed_W_1;
    xt::xarray<double> smoothed_W_2;
    std::unique_ptr<AbstractActivation> afunc_1;
    std::unique_ptr<AbstractActivation> afunc_2;

public:
    explicit SpikeDNNet(
        std::unique_ptr<AbstractActivation> act_func_1,
        std::unique_ptr<AbstractActivation> act_func_2,
        xt::xarray<double> mat_W_1,
        xt::xarray<double> mat_W_2,
        size_t dim = 2,
        xt::xarray<double> mat_A = 20. * xt::diag(xt::xarray<double>({-1., -2.})),
        xt::xarray<double> mat_P = 1575.9 * xt::diag(xt::xarray<double>({60., 40.})),
        xt::xarray<double> mat_K_1 = .15 * xt::diag(xt::xarray<double>({10., 1.})),
        xt::xarray<double> mat_K_2 = .15 * xt::diag(xt::xarray<double>({1., 1.}))
    );

    SpikeDNNet(const SpikeDNNet& other) noexcept;

    SpikeDNNet& operator=(const SpikeDNNet& other) noexcept;

    static xt::xarray<double> moving_average(xt::xarray<double> x, std::uint32_t w = 2);

    xt::xarray<double> smooth(xt::xarray<double> x, std::uint32_t w = 2);

    xt::xarray<double> fit(
        xt::xarray<double> vec_x,
        xt::xarray<double> vec_u,
        double step = .01,
        std::uint32_t n_epochs = 3,
        std::uint32_t k_points = 2);

    xt::xarray<double> predict(
        xt::xarray<double> init_state,
        xt::xarray<double> vec_u,
        double step = .01);

    xt::xarray<double> get_weights(std::uint8_t idx) const;
};

};