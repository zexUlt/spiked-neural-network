#include "SpikeDNNet.hpp"
#include "IzhikevichActivation.hpp"
#include "SigmoidActivation.hpp"
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
    auto tr_target  = xt::load_npy<double>("../train_data/tr_target.npy");
    auto tr_control = xt::load_npy<double>("../train_data/tr_control.npy");
    auto vl_target  = xt::load_npy<double>("../train_data/vl_target.npy");
    auto vl_control = xt::load_npy<double>("../train_data/vl_control.npy");

    const std::uint32_t width    = 4394u;
    const std::int32_t split     = 3305;
    const std::uint32_t dim      = 2u;
    const std::uint32_t n_epochs = 1u;
    const std::uint32_t k_points = 1u;


    auto izh_act_1 = new CxxSDNN::IzhikevichActivation(dim);
    auto izh_act_2 = new CxxSDNN::IzhikevichActivation(dim);
    // auto sigm_act_1 = new CxxSDNN::SigmoidActivation();
    // auto sigm_act_2 = new CxxSDNN::SigmoidActivation();


    auto W_1 = 1. * xt::ones<double>({dim, dim})  ;
    auto W_2 = 1. * xt::ones<double>({dim, dim});
    auto A   = 162. * xt::diag(xt::xarray<double>{-1., -1.});
    auto P   = 3337 * xt::diag(xt::xarray<double>{1., 1.});
    auto K_1 = 1 * xt::diag(xt::xarray<double>{1., 1.});
    auto K_2 = 0.1 * xt::diag(xt::xarray<double>{1., 1.});  

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
    
    auto error = xt::abs(xt::col(tr_target, 0) - xt::col(res.tr_est[0], 1));
    // auto wdiff1 = xt::diff(res.W_1)
    xt::dump_npy("../plot_data/error.npy", xt::degrees(error));
    xt::dump_npy("../plot_data/control.npy", xt::degrees(tr_control));  
    xt::dump_npy("../plot_data/target.npy", xt::degrees(xt::col(tr_target, 0)));
    xt::dump_npy("../plot_data/estimation.npy", xt::degrees(xt::col(res.tr_est[0], 0)));
    xt::dump_npy("../plot_data/target2.npy", xt::degrees(xt::col(tr_target, 1)));
    xt::dump_npy("../plot_data/estimation2.npy", xt::degrees(xt::col(res.tr_est[0], 1)));

    delete izh_act_1;
    delete izh_act_2;
    return 0;
}