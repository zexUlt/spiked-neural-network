#pragma once
// #include "precompiled.hpp"
#include <string>
#include <vector>

namespace cxx_sdnn
{
  namespace optimization
  {
    struct TrainedParams
    {
      double a;
      double p;
      double k1;
      double k2;
      double alpha;
      double sigm_a;
      double sigm_b;
      double sigm_c;
      double sigm_d;
      double sigm_e;
    };

    auto
    setup_dnn(std::uint32_t targetDim, std::uint32_t controlDim, const TrainedParams& params, std::string neuronType);
    double estimate_loss(std::string trainingDataRoot, TrainedParams params, std::string neuronType);
    double run_minimize(std::vector<double> intialParams, std::string neuronType, std::string trainingDataRoot);
    void dummy();
  }; // namespace optimization
};   // namespace cxx_sdnn
