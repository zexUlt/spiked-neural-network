#pragma once
// #include "precompiled.hpp"
#include "string"

namespace cxx_sdnn::optimization{
    struct TrainedParams{
        double a;
        double p;
        double k1;
        double k2;
        double alpha;
    };

    auto setup_dnn(std::uint32_t targetDim, std::uint32_t controlDim, const TrainedParams& params, std::string neuronType);
    double estimate_loss(std::string trainingDataRoot, TrainedParams params, std::string neuronType);
    double run_minimize(double a, double p, double k1, double k2, double alpha, std::string neuronType);
    void dummy();
}; // namespace cxx_sdnn::optimization
