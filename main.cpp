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
    using namespace xt::placeholders;

    xt::xarray<double> tr_raw = xt::load_npy<double>("../train_data/tr_target.npy");
    xt::xarray<double> tr_target_coord = xt::view(tr_raw, xt::range(1, _));
    xt::xarray<double> tr_target_speed = xt::diff(tr_raw, 1, 0) / 120.;
    xt::xarray<double> tr_target = xt::concatenate(xt::xtuple(tr_target_coord, tr_target_speed), 1);

    xt::xarray<double> vl_raw  = xt::load_npy<double>("../train_data/vl_target.npy");
    xt::xarray<double> vl_target_coord = xt::view(vl_raw, xt::range(1, _));
    xt::xarray<double> vl_target_speed = xt::diff(vl_raw, 1, 0) / 120.;
    xt::xarray<double> vl_target = xt::concatenate(xt::xtuple(vl_target_coord, vl_target_speed), 1);
    xt::xarray<double> tr_control = xt::diff(xt::load_npy<double>("../train_data/tr_control.npy"), 1, 0) / 120.;
    xt::xarray<double> vl_control = xt::diff(xt::load_npy<double>("../train_data/vl_control.npy"), 1, 0) / 120.;

    const std::uint32_t width    = 4394u;
    const std::int32_t split     = 3305;
    const std::uint32_t dim      = 4u;
    const std::uint32_t n_epochs = 2u;
    const std::uint32_t k_points = 3u;

    using Izhi = CxxSDNN::IzhikevichActivation;

    auto izh_act_1 = UtilityFunctionLibrary::make_izhikevich(50, 1/60., {tr_target.shape(1), 1}, Izhi::NeuronType::RegularSpiking);
    auto izh_act_2 = UtilityFunctionLibrary::make_izhikevich(50, 1/60., {tr_target.shape(1), tr_control.shape(1)}, Izhi::NeuronType::RegularSpiking);
    // auto sigm_act_1 = std::make_unique<CxxSDNN::SigmoidActivation>();
    // auto sigm_act_2 = std::make_unique<CxxSDNN::SigmoidActivation>();

    auto W_1 = 1. * xt::ones<double>({dim, dim}); // 1.
    auto W_2 = 1. * xt::ones<double>({dim, dim}); // 1.
    auto A   = 162. * xt::diag(xt::xarray<double>{-1., -1., -1., -1.}); // 162.
    auto P   = 3337. * xt::diag(xt::xarray<double>{1., 1., 1., 1.}); // 3337.
    auto K_1 = .5 * xt::diag(xt::xarray<double>{1., 1., 1., 1.}); // 1.
    auto K_2 = 1. * xt::diag(xt::xarray<double>{1., 1., 1., 1.});  // 0.1

    CxxSDNN::SpikeDNNet dnn_izh(
        std::move(izh_act_1), std::move(izh_act_2), // Activation functions
        W_1, W_2, // W_1, W_2
        dim, A, // dim, mat_A
        P, K_1, // mat_P, K_1
        K_2 // K_2
    );

    // CxxSDNN::SpikeDNNet dnn_sigm(
    //     std::move(sigm_act_1), std::move(sigm_act_2), // Activation functions
    //     W_1, W_2, // W_1, W_2
    //     dim, A, // dim, mat_A
    //     P, K_1, // mat_P, K_1
    //     K_2 // K_2
    // );

    UtilityFunctionLibrary::vl_tr_map<double> folds{
        {
            "tr", {tr_target, tr_control}
        }, 
        {
            "vl", {vl_target, vl_control}
        }
    };

    std::cout << "Izhikevich timings: " << UtilityFunctionLibrary::timeit([&dnn_izh, &tr_target, &tr_control, &n_epochs, &k_points](){ 
        dnn_izh.fit(tr_target, tr_control, 0.01, n_epochs, k_points); }, 100u) << '\n';
    
    // std::cout << "Sigmoidal timings: " << UtilityFunctionLibrary::timeit([&dnn_sigm, &tr_target, &tr_control, &n_epochs, &k_points](){ 
    //     dnn_sigm.fit(tr_target, tr_control, 0.01, n_epochs, k_points); }, 100u) << '\n';

    // auto res = UtilityFunctionLibrary::dnn_validate(dnn_izh, folds, n_epochs, k_points);

    // std::cout << res;

    // UtilityFunctionLibrary::dumpData(tr_target, tr_control, res);

    return 0;
}