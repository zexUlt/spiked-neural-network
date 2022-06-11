#include "IzhikevichActivation.hpp"
#include "SigmoidActivation.hpp"
#include "SpikeDNNet.hpp"
#include "Utility.hpp"
#include "debug_header.hpp"

#include <fstream>
#include <istream>
#include <string>
#include <unordered_map>
#include <xtensor/xarray.hpp>
#include <xtensor/xio.hpp>

int main(int argc, char** argv)
{
  using namespace xt::placeholders;

  xt::xarray<double> trRaw         = xt::load_npy<double>("../train_data/tr_target.npy");
  xt::xarray<double> trTargetCoord = xt::view(trRaw, xt::range(1, _));
  xt::xarray<double> trTargetSpeed = xt::diff(trRaw, 1, 0) / 120.;
  xt::xarray<double> trTarget      = xt::concatenate(xt::xtuple(trTargetCoord, trTargetSpeed), 1);

  xt::xarray<double> vlRaw         = xt::load_npy<double>("../train_data/vl_target.npy");
  xt::xarray<double> vlTargetCoord = xt::view(vlRaw, xt::range(1, _));
  xt::xarray<double> vlTargetSpeed = xt::diff(vlRaw, 1, 0) / 120.;
  xt::xarray<double> vlTarget      = xt::concatenate(xt::xtuple(vlTargetCoord, vlTargetSpeed), 1);
  xt::xarray<double> trControl     = xt::diff(xt::load_npy<double>("../train_data/tr_control.npy"), 1, 0) / 120.;
  xt::xarray<double> vlControl     = xt::diff(xt::load_npy<double>("../train_data/vl_control.npy"), 1, 0) / 120.;

  // const std::uint32_t width    = 4394u;
  // const std::int32_t split     = 3305;
  const std::uint32_t N_EPOCHS = 6u;
  const std::uint32_t K_POINTS = 3u;

  std::uint32_t dim = trTarget.shape(1);

  std::unordered_map<std::string, xt::xarray<double>> modelParams{
    {"W_1", -.1 * xt::ones<double>({dim, dim})},
    {"W_2", -.1 * xt::ones<double>({dim, dim})},
    {"A", 160. * xt::diag(xt::xarray<double>{-1., -1., -1., -1.})},
    {"P", 50. * xt::diag(xt::xarray<double>{1., 1., 1., 1.})},
    {"K_1", 3000. * xt::diag(xt::xarray<double>{1., 1., 1., 1.})},
    {"K_2", .001 * xt::diag(xt::xarray<double>{1., 1., 1., 1.})}};

  // auto izh_act_1 = UtilityFunctionLibrary::make_izhikevich(50, 1/40., {2*dim, 1}, Izhi::NeuronType::Resonator);
  // auto izh_act_2 = UtilityFunctionLibrary::make_izhikevich(55, 1/40., {2*dim, tr_control.shape(1)},
  // Izhi::NeuronType::ThalamoCortical63);

  auto act1 = std::make_unique<cxx_sdnn::SigmoidActivation>(/*shape=*/std::vector<size_t>{dim, 1});
  auto act2 = std::make_unique<cxx_sdnn::SigmoidActivation>(/*shape=*/std::vector<size_t>{dim, trControl.shape(1)});

  auto model = UtilityFunctionLibrary::make_dnn(dim, std::move(act1), std::move(act2), modelParams);

  UtilityFunctionLibrary::VlTrMap<double> folds{{"tr", {trTarget, trControl}}, {"vl", {vlTarget, vlControl}}};

  auto res = UtilityFunctionLibrary::dnn_validate(std::move(model), folds, N_EPOCHS, K_POINTS);

  std::cout << res;

  UtilityFunctionLibrary::dump_data(trTarget, trControl, res);

  return 0;
}