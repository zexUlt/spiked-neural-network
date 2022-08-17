#include "Optimization.hpp"

#include "IzhikevichActivation.hpp"
#include "SigmoidActivation.hpp"
#include "SpikeDNNet.hpp"
#include "Utility/Utility.hpp"
#include "debug_header.hpp"

#include "xtensor/xarray.hpp"

#include <memory>
#include <vector>

namespace cxx_sdnn
{
  namespace optimization
  {
    auto
    setup_dnn(std::uint32_t targetDim, std::uint32_t controlDim, const TrainedParams& params, std::string neuronType)
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
        auto izhAct1 = cxx_sdnn::make_izhikevich(50, 1 / 40., {2 * targetDim, 1}, Izhi::NeuronType::RESONATOR);
        auto izhAct2 =
          cxx_sdnn::make_izhikevich(55, 1 / 40., {2 * targetDim, controlDim}, Izhi::NeuronType::THALAMO_CORTICAL63);

        izhAct1->set_integration_step(0.0001);
        izhAct2->set_integration_step(0.0001);

        auto dnn = cxx_sdnn::UtilityFunctionLibrary::make_dnn(
          params.alpha, targetDim, std::move(izhAct1), std::move(izhAct2), modelParams);

        return dnn;
      } else {
        auto sigmAct1 = std::make_unique<cxx_sdnn::SigmoidActivation>(
          /*shape=*/std::vector<size_t>{targetDim, 1},
          /*a =*/params.sigm_a,
          /*b =*/params.sigm_b,
          /*c =*/params.sigm_c,
          /*d =*/params.sigm_d,
          /*e =*/params.sigm_e);

        auto sigmAct2 = std::make_unique<cxx_sdnn::SigmoidActivation>(
          /*shape=*/std::vector<size_t>{targetDim, controlDim},
          /*a =*/params.sigm_a,
          /*b =*/params.sigm_b,
          /*c =*/params.sigm_c,
          /*d =*/params.sigm_d,
          /*e =*/params.sigm_e);
          
        auto dnn = cxx_sdnn::UtilityFunctionLibrary::make_dnn(
          params.alpha, targetDim, std::move(sigmAct1), std::move(sigmAct2), modelParams);

        return dnn;
      }
    }

    double estimate_loss(std::string trainingDataRoot, TrainedParams params, std::string neuronType)
    {
      static auto trainData  = cxx_sdnn::UtilityFunctionLibrary::prepare_dataset(trainingDataRoot)["tr"];
      static auto targetDim  = trainData.first.shape(1);
      static auto controlDim = trainData.second.shape(1);

      const double STEP            = 0.01;
      const std::uint32_t EPOCHS   = 1u;
      const std::uint32_t K_POINTS = 2u;
      auto dnn                     = setup_dnn(targetDim, controlDim, params, neuronType);

      dnn->fit(trainData.first, trainData.second, STEP, EPOCHS, K_POINTS);

      return dnn->integral_loss();
    }

    double run_minimize(std::vector<double> initialParams, std::string neuronType, std::string trainingDataRoot)
    {
      TrainedParams params{
        initialParams[0], initialParams[1], initialParams[2], initialParams[3], initialParams[4],
        initialParams[5], initialParams[6], initialParams[7], initialParams[8], initialParams[9]};
      return estimate_loss(trainingDataRoot, params, neuronType);
    }

    void dummy()
    {
      std::cout << "Hello World!\n";
    }
  } // namespace optimization
};  // namespace cxx_sdnn