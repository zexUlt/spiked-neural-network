#include "Optimization.hpp"

#include "IzhikevichActivation.hpp"
#include "SigmoidActivation.hpp"
#include "SpikeDNNet.hpp"
#include "Utility.hpp"
#include "vector"

#include "xtensor/xarray.hpp"

namespace cxx_sdnn::optimization
{
  auto setup_dnn(std::uint32_t targetDim, std::uint32_t controlDim, const TrainedParams& params, std::string neuronType)
  {
    using Izhi = IzhikevichActivation;

    std::unordered_map<std::string, xt::xarray<double>> modelParams{
      {"W_1", 1000. * xt::ones<double>({targetDim, targetDim})},
      {"W_2", 1000. * xt::ones<double>({targetDim, targetDim})},
      {"A", params.a * xt::diag(xt::xarray<double>{-1., -1., -1., -1.})},
      {"P", params.p * xt::diag(xt::xarray<double>{1., 1., 1., 1.})},
      {"K_1", params.k1 * xt::diag(xt::xarray<double>{1., 1., 1., 1.})},
      {"K_2", params.k2 * xt::diag(xt::xarray<double>{1., 1., 1., 1.})}};

    if(neuronType == "izhikevich") {
      auto izhAct1 = make_izhikevich(50, 1 / 40., {2 * targetDim, 1}, Izhi::NeuronType::RESONATOR);
      auto izhAct2 = make_izhikevich(55, 1 / 40., {2 * targetDim, controlDim}, Izhi::NeuronType::THALAMO_CORTICAL63);

      auto dnn =
        UtilityFunctionLibrary::make_dnn(params.alpha, targetDim, std::move(izhAct1), std::move(izhAct2), modelParams);

      return dnn;
    } else {
      auto sigmAct1 = std::make_unique<cxx_sdnn::SigmoidActivation>(
        /*shape=*/std::vector<size_t>{targetDim, 1},
        /*a =*/1.,
        /*b =*/1.,
        /*c =*/1,
        /*d =*/-1,
        /*e =*/-0.5);

      auto sigmAct2 = std::make_unique<cxx_sdnn::SigmoidActivation>(
        /*shape=*/std::vector<size_t>{targetDim, controlDim},
        /*a =*/1.,
        /*b =*/1.,
        /*c =*/1.,
        /*d =*/-1,
        /*e =*/-0.5);

      auto dnn = UtilityFunctionLibrary::make_dnn(
        params.alpha, targetDim, std::move(sigmAct1), std::move(sigmAct2), modelParams);

      return dnn;
    }
  }

  double estimate_loss(std::string trainingDataRoot, TrainedParams params, std::string neuronType)
  {
    static auto trainData  = UtilityFunctionLibrary::prepare_dataset(trainingDataRoot)["tr"];
    static auto targetDim  = trainData.first.shape(1);
    static auto controlDim = trainData.second.shape(1);

    const double STEP            = 0.01;
    const std::uint32_t EPOCHS   = 1u;
    const std::uint32_t K_POINTS = 2u;
    auto dnn                     = setup_dnn(targetDim, controlDim, params, neuronType);

    dnn->fit(trainData.first, trainData.second, STEP, EPOCHS, K_POINTS);

    return dnn->integral_loss();
  }

  double run_minimize(double a, double p, double k1, double k2, double alpha, std::string neuronType)
  {
    TrainedParams params{a, p, k1, k2, alpha};
    return estimate_loss("", params, neuronType);
  }

  void dummy()
  {
    std::cout << "Hello World!\n";
  }
}; // namespace cxx_sdnn::optimization