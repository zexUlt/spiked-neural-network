#include "SpikeDNNet.hpp"
#include "IzhikevichActivation.hpp"
#include "SigmoidActivation.hpp"
#include "Utility.hpp"

#include <xtensor/xarray.hpp>
#include <xtensor/xio.hpp>

#include <istream>
#include <fstream>
#include <unordered_map>
#include <string>

#include "debug_header.hpp"


int main(int argc, char** argv)
{   
    using namespace xt::placeholders;

    xt::xarray<double> tr_raw          = xt::load_npy<double>("../train_data/tr_target.npy");
    xt::xarray<double> tr_target_coord = xt::view(tr_raw, xt::range(1, _));
    xt::xarray<double> tr_target_speed = xt::diff(tr_raw, 1, 0) / 120.;
    xt::xarray<double> tr_target       = xt::concatenate(xt::xtuple(tr_target_coord, tr_target_speed), 1);

    xt::xarray<double> vl_raw          = xt::load_npy<double>("../train_data/vl_target.npy");
    xt::xarray<double> vl_target_coord = xt::view(vl_raw, xt::range(1, _));
    xt::xarray<double> vl_target_speed = xt::diff(vl_raw, 1, 0) / 120.;
    xt::xarray<double> vl_target       = xt::concatenate(xt::xtuple(vl_target_coord, vl_target_speed), 1);
    xt::xarray<double> tr_control      = xt::diff(xt::load_npy<double>("../train_data/tr_control.npy"), 1, 0) / 120.;
    xt::xarray<double> vl_control      = xt::diff(xt::load_npy<double>("../train_data/vl_control.npy"), 1, 0) / 120.;

    // const std::uint32_t width    = 4394u;
    // const std::int32_t split     = 3305;
    const std::uint32_t n_epochs = 6u;
    const std::uint32_t k_points = 3u;

    std::uint32_t dim            = tr_target.shape(1);

    std::unordered_map<std::string, xt::xarray<double>> model_params{
        {"W_1", -.1 * xt::ones<double>({dim, dim})},
        {"W_2", -.1 * xt::ones<double>({dim, dim})},
        {"A", 160. * xt::diag(xt::xarray<double>{-1., -1., -1., -1.})},
        {"P", 50. * xt::diag(xt::xarray<double>{1., 1., 1., 1.})},
        {"K_1", 3000. * xt::diag(xt::xarray<double>{1., 1., 1., 1.})},
        {"K_2", .001 * xt::diag(xt::xarray<double>{1., 1., 1., 1.})}        
    };

    // auto izh_act_1 = UtilityFunctionLibrary::make_izhikevich(50, 1/40., {2*dim, 1}, Izhi::NeuronType::Resonator);
    // auto izh_act_2 = UtilityFunctionLibrary::make_izhikevich(55, 1/40., {2*dim, tr_control.shape(1)}, Izhi::NeuronType::ThalamoCortical63);

    auto act_1 = std::make_unique<cxx_sdnn::SigmoidActivation>(std::vector<size_t>{dim, 1});
    auto act_2 = std::make_unique<cxx_sdnn::SigmoidActivation>(std::vector<size_t>{dim, tr_control.shape(1)});

    auto model = UtilityFunctionLibrary::make_dnn(dim, std::move(act_1), std::move(act_2), model_params);

    UtilityFunctionLibrary::VlTrMap<double> folds{
        {
            "tr", {tr_target, tr_control}
        }, 
        {
            "vl", {vl_target, vl_control}
        }
    };

    auto res = UtilityFunctionLibrary::dnn_validate(std::move(model), folds, n_epochs, k_points);

    std::cout << res;

    UtilityFunctionLibrary::dump_data(tr_target, tr_control, res);

    return 0;
}