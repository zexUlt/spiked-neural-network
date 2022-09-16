#pragma once

#include "IzhikevichActivation.hpp"
#include "SpikeDNNet.hpp"
#include "precompiled.hpp"

namespace cxx_sdnn
{
  class UtilityFunctionLibrary {
  public:
    using VlTrMap = std::map<std::string, std::pair<xt::xarray<double>, xt::xarray<double>>>;
    using Izhi    = cxx_sdnn::IzhikevichActivation;

    struct ValidationResults
    {
      std::pair<double, double> mseRes;
      std::pair<double, double> maeRes;
      std::vector<xt::xarray<double>> trEst;
      std::vector<xt::xarray<double>> vlRes;
      xt::xarray<double> w1;
      xt::xarray<double> w2;
      xt::xarray<double> n1;
      xt::xarray<double> n2;

      friend std::ostream& operator<<(std::ostream& out, UtilityFunctionLibrary::ValidationResults res)
      {
        out << "MSE: [Train: " << res.mseRes.first << ", Test: " << res.mseRes.second << "]\n";
        out << "MAE: [Train: " << res.maeRes.first << ", Test: " << res.maeRes.second << "]\n";

        return out;
      }
    };

    static xt::xarray<double> convolve_valid(const xt::xarray<double>& f, const xt::xarray<double>& g);

    static ValidationResults dnn_validate(
      std::unique_ptr<cxx_sdnn::SpikeDNNet> dnn, VlTrMap folds, std::uint16_t nEpochs, std::uint16_t kPoints,
      double step = 0.01);

    static double mean_squared_error(xt::xarray<double> yTrue, xt::xarray<double> yPred);

    static double mean_absolute_error(xt::xarray<double> yTrue, xt::xarray<double> yPred);

    static void dump_data(std::string&& plotDataExportRoot, xt::xarray<double> trTarget, xt::xarray<double> trControl, ValidationResults data);

    static double timeit(std::function<void(void)> foo, std::uint32_t count = 1);

    static std::unique_ptr<cxx_sdnn::SpikeDNNet> make_dnn(
      double alpha, std::uint32_t dim, std::unique_ptr<cxx_sdnn::AbstractActivation> act1,
      std::unique_ptr<cxx_sdnn::AbstractActivation> act2, std::unordered_map<std::string, xt::xarray<double>>& kwargs);

    static VlTrMap prepare_dataset(std::string trainingDataImportRoot);
  };
}; // namespace cxx_sdnn