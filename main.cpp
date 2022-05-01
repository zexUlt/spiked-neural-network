#include "SpikeDNNet.hpp"
#include "IzhikevichActivation.hpp"
#include "SigmoidActivation.hpp"
#include "Utility.hpp"

#include <xtensor/xarray.hpp>
#include <xtensor/xio.hpp>

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

    std::unique_ptr<CxxSDNN::IzhikevichActivation> izh_act_1 = std::make_unique<CxxSDNN::IzhikevichActivation>(dim);
    std::unique_ptr<CxxSDNN::IzhikevichActivation> izh_act_2 = std::make_unique<CxxSDNN::IzhikevichActivation>(dim);


    auto W_1 = .01 * xt::ones<double>({dim, dim}); // 1.
    auto W_2 = 10. * xt::ones<double>({dim, dim}); // 1.
    auto A   = 162. * xt::diag(xt::xarray<double>{-1., -1.}); // 162.
    auto P   = 3337. * xt::diag(xt::xarray<double>{1., 1.}); // 3337.
    auto K_1 = 1. * xt::diag(xt::xarray<double>{1., 1.}); // 1.
    auto K_2 = .1 * xt::diag(xt::xarray<double>{1., 1.});  // 0.1

    CxxSDNN::SpikeDNNet dnn(
        std::move(izh_act_1), std::move(izh_act_2), // Activation functions
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

    UtilityFunctionLibrary::dumpData(tr_target, tr_control, res);

    return 0;
}