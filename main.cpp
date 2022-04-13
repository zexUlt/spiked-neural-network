#include "SpikeDNNet.hpp"
#include "IzhikevichActivation.hpp"
#include "Utility.hpp"

#include <xtensor/xarray.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xview.hpp>
#include <xtensor/xnpy.hpp>

#include <xtensor-blas/xlinalg.hpp>

#include <istream>
#include <fstream>

#include "debug_header.hpp"

int main(int argc, char** argv)
{
    // xt::xarray<int> a{1,2,3};
    // double av = xt::average(a)();
    // std::cout << av;

    // std::cout << a << "\n";
    // auto av = xt::view(a, xt::all(), 0, 0);
    // av.assign(xt::arange<int>(0, 3));
    // std::cout << a << '\n';

    auto tr_target  = xt::load_npy<double>("../train_data/tr_target.npy");
    auto tr_control = xt::load_npy<double>("../train_data/tr_control.npy");
    auto vl_target  = xt::load_npy<double>("../train_data/vl_target.npy");
    auto vl_control = xt::load_npy<double>("../train_data/vl_control.npy");

    // auto tr_target  = nc::fromfile<double>("train_data/tr_target.dmp").reshape(-1, 2);
    // auto tr_control = nc::fromfile<double>("train_data/tr_control.dmp").reshape(-1, 2);
    // auto vl_target  = nc::fromfile<double>("train_data/vl_target.dmp").reshape(-1, 2);
    // auto vl_control = nc::fromfile<double>("train_data/vl_control.dmp").reshape(-1, 2);

    const std::uint32_t width    = 4394u;
    const std::int32_t split     = 3305;
    const std::uint32_t dim      = 2u;
    const std::uint32_t n_epochs = 1u;
    const std::uint32_t k_points = 1u;


    auto izh_act_1 = new CxxSDNN::IzhikevichActivation(dim);
    auto izh_act_2 = new CxxSDNN::IzhikevichActivation(dim);

    auto W_1 = 20. * xt::ones<double>({dim, dim})  ;
    auto W_2 = 20. * xt::ones<double>({dim, dim});
    auto A   = 20. * xt::diag(xt::xarray<double>{-1., -2.});
    auto P   = 1575.9 * xt::diag(xt::xarray<double>{60., 40,});
    auto K_1 = .15 * xt::diag(xt::xarray<double>{10., 1.});
    auto K_2 = .15 * xt::diag(xt::xarray<double>{1., 1.});  

    CxxSDNN::SpikeDNNet dnn(
        izh_act_1, izh_act_2, // Activation functions
        W_1, W_2, // W_1, W_2
        dim, A, // dim, mat_A
        P, K_1, // mat_P, K_1
        K_2 // K_2
    );

    UtilityFunctionLibrary::vl_tr_map<double> folds{
        {
            "tr", {tr_target, tr_control}
        }, 
        {
            "vl", {vl_target, vl_control}
        }
    };

    auto res = UtilityFunctionLibrary::dnn_validate(dnn, folds, n_epochs, k_points);

    std::cout << res;

    auto error = xt::abs(xt::col(tr_target, 0) - xt::col(res.tr_est[0], 0));

    xt::dump_npy("../plot_data/error.npy", error);
    xt::dump_npy("../plot_data/target.npy", xt::degrees(xt::col(tr_target, 0)) + 2.);
    xt::dump_npy("../plot_data/control.npy", xt::degrees(tr_control));
    xt::dump_npy("../plot_data/estimation.npy", xt::degrees(xt::col(res.tr_est[0], 0)) + 2.);

    return 0;
}