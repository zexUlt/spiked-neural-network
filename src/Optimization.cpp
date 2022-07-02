#include "Optimization.hpp"
#include "SpikeDNNet.hpp"
#include "Utility.hpp"
#include "SigmoidActivation.hpp"
#include "IzhikevichActivation.hpp"

namespace cxx_sdnn::optimization{
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

        if(neuronType == "izhikevich"){
            auto izh_act_1 = make_izhikevich(50, 1/40., {2*targetDim, 1}, Izhi::NeuronType::RESONATOR);
            auto izh_act_2 = make_izhikevich(55, 1/40., {2*targetDim, controlDim}, Izhi::NeuronType::THALAMO_CORTICAL63);
    
            auto dnn = UtilityFunctionLibrary::make_dnn(params.alpha, targetDim, std::move(izh_act_1), std::move(izh_act_2), modelParams);

            return dnn;
        }else{
            auto sigm_act_1 = std::make_unique<cxx_sdnn::SigmoidActivation>(
            /*shape=*/std::vector<size_t>{targetDim, 1},
            /*a =*/1.,
            /*b =*/1.,
            /*c =*/1,
            /*d =*/-1,
            /*e =*/-0.5);
            
            auto sigm_act_2 = std::make_unique<cxx_sdnn::SigmoidActivation>(
            /*shape=*/std::vector<size_t>{targetDim, controlDim},
            /*a =*/1.,
            /*b =*/1.,
            /*c =*/1.,
            /*d =*/-1,
            /*e =*/-0.5);
            
            auto dnn = UtilityFunctionLibrary::make_dnn(params.alpha, targetDim, std::move(sigm_act_1), std::move(sigm_act_2), modelParams);

            return dnn;
        }
    }

    double estimate_loss(const TrainedParams& params, std::string neuronType)
    {
        static auto trainData = UtilityFunctionLibrary::prepare_dataset()["tr"];
        static auto targetDim = trainData.first.shape(1);
        static auto controlDim = trainData.second.shape(1);

        const double step = 0.01;
        const std::uint32_t epochs = 1u;
        const std::uint32_t kPoints = 2u;
        auto dnn = setup_dnn(targetDim, controlDim, params, neuronType);
        
        dnn->fit(trainData.first, trainData.second, step, epochs, kPoints);

        return dnn->integral_loss();
    }

    double run_minimize(double a, double p, double k1, double k2, double alpha, std::string neuronType)
    {
        TrainedParams params{a, p, k1, k2, alpha};
        return estimate_loss(params, neuronType);
    }
};