#include "IzhikevichActivation.hpp"
#include "SigmoidActivation.hpp"
#include "SpikeDNNet.hpp"
#include "Utility.hpp"
#include "debug_header.hpp"
#include "precompiled.hpp"

int main(int argc, char** argv)
{
  auto folds = UtilityFunctionLibrary::prepare_dataset();

  // const std::uint32_t width    = 4394u;
  // const std::int32_t split     = 3305;

  const std::uint32_t N_EPOCHS  = 2u;
  const std::uint32_t K_POINTS  = 3u;
  const double ALPHA            = 1.5;
  const double INTEGRATION_STEP = 0.0001;

  std::uint32_t targetDim = folds["tr"].first.shape(1);
  std::uint32_t controlDim = folds["tr"].second.shape(1);

  std::unordered_map<std::string, xt::xarray<double>> modelParams{
    {"W_1", 1000. * xt::ones<double>({targetDim, targetDim})},
    {"W_2", 1000. * xt::ones<double>({targetDim, targetDim})},
    {"A", 10000 * xt::diag(xt::xarray<double>{-1., -1., -1., -1.})},
    {"P", 9000 * xt::diag(xt::xarray<double>{1., 1., 1., 1.})},
    {"K_1", 3000 * xt::diag(xt::xarray<double>{1., 1., 1., 1.})},
    {"K_2", 1000 * xt::diag(xt::xarray<double>{1., 1., 1., 1.})}};

  // auto izh_act_1 = UtilityFunctionLibrary::make_izhikevich(50, 1/40., {2*targetDim, 1}, Izhi::NeuronType::Resonator);
  // auto izh_act_2 = UtilityFunctionLibrary::make_izhikevich(55, 1/40., {2*targetDim, controlDim},
  // Izhi::NeuronType::ThalamoCortical63);

  auto act1 = std::make_unique<cxx_sdnn::SigmoidActivation>(
    /*shape=*/std::vector<size_t>{targetDim, 1},
    /*a =*/1.,
    /*b =*/1.,
    /*c =*/1,
    /*d =*/-1,
    /*e =*/-0.5);
  auto act2 = std::make_unique<cxx_sdnn::SigmoidActivation>(
    /*shape=*/std::vector<size_t>{targetDim, controlDim},
    /*a =*/1.,
    /*b =*/1.,
    /*c =*/1.,
    /*d =*/-1,
    /*e =*/-0.5);

  auto model =
    UtilityFunctionLibrary::make_dnn(ALPHA, targetDim, std::move(act1), std::move(act2), modelParams);

  auto res = UtilityFunctionLibrary::dnn_validate(std::move(model), folds, N_EPOCHS, K_POINTS, INTEGRATION_STEP);

  std::cout << res;

  UtilityFunctionLibrary::dump_data(trTarget, trControl, res);

  return 0;
}