#pragma once

#include "IzhikevichActivation.hpp"
#include "SpikeDNNet.hpp"
#include "debug_header.hpp"
#include "precompiled.hpp"

class UtilityFunctionLibrary {
public:
  template<typename Dtype>
  using VlTrMap = std::map<std::string, std::pair<xt::xarray<Dtype>, xt::xarray<Dtype>>>;
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

  static xt::xarray<double> convolve_valid(const xt::xarray<double>& f, const xt::xarray<double>& g)
  {
    const auto NF = f.size();
    const auto NG = g.size();

    const auto& minV = (NF < NG) ? f : g;
    const auto& maxV = (NF < NG) ? g : f;
    const auto N     = std::max(NF, NG) - std::min(NF, NG) + 1;

    xt::xarray<double> out = xt::zeros<double>({N});

    for(auto i(0u); i < N; ++i) {
      for(int j(minV.size() - 1), k(i); j >= 0; --j, ++k) {
        out.at(i) += minV[j] * maxV[k];
      }
    }

    return out;
  }

  static ValidationResults dnn_validate(
    std::unique_ptr<cxx_sdnn::SpikeDNNet> dnn, VlTrMap<double> folds, std::uint16_t nEpochs, std::uint16_t kPoints)
  {
    ValidationResults results;

    std::cout << *dnn;

    auto trTarget  = folds["tr"].first / 10.;
    auto trControl = folds["tr"].second;

    auto vlTarget  = folds["vl"].first;
    auto vlControl = folds["vl"].second;

    auto targetEst = dnn->fit(trTarget, trControl, 0.0001, nEpochs, kPoints);
    auto vlPred    = dnn->predict(xt::view(trTarget, -1, 0), vlControl);

    xt::xarray<double> trCol     = xt::col(trTarget, 0);
    xt::xarray<double> targetCol = xt::col(targetEst, 0);
    xt::xarray<double> vlCol     = xt::col(vlTarget, 0);
    xt::xarray<double> predCol   = xt::col(vlPred, 0);
    results.mseRes               = std::make_pair(
      UtilityFunctionLibrary::mean_squared_error<double>(trCol, targetCol),
      UtilityFunctionLibrary::mean_squared_error<double>(vlCol, predCol));

    results.maeRes = std::make_pair(
      UtilityFunctionLibrary::mean_absolute_error<double>(trCol, targetCol),
      UtilityFunctionLibrary::mean_absolute_error<double>(vlCol, predCol));

    results.trEst.emplace_back(targetEst);
    results.vlRes.emplace_back(vlPred);
    results.w1 = dnn->get_weights(0);
    results.w2 = dnn->get_weights(1);
    results.n1 = dnn->get_neurons_history(0);
    results.n2 = dnn->get_neurons_history(1);

    return results;
  }

  template<typename Dtype>
  static double mean_squared_error(xt::xarray<Dtype> yTrue, xt::xarray<Dtype> yPred)
  {
    xt::xarray<Dtype> sq = xt::square(yTrue - yPred);
    auto outputErrors    = xt::average(sq);

    return outputErrors();
  }

  template<typename Dtype>
  static double mean_absolute_error(xt::xarray<Dtype> yTrue, xt::xarray<Dtype> yPred)
  {
    xt::xarray<Dtype> absol = xt::abs(yPred - yTrue);
    auto outputErrors       = xt::average(absol);

    return outputErrors();
  }

  static void dump_data(xt::xarray<double> trTarget, xt::xarray<double> trControl, ValidationResults data)
  {
    auto error  = xt::abs(xt::col(trTarget, 0) - xt::col(data.trEst[0], 0));
    auto wdiff1 = xt::view(data.w1, xt::all(), xt::all(), 1);
    auto wdiff2 = xt::view(data.w2, xt::all(), xt::all(), 1);

    std::cout << "Error: " << xt::average(error) << "\n";

    xt::dump_npy("../plot_data/error.npy", xt::degrees(error));
    xt::dump_npy("../plot_data/control.npy", xt::degrees(trControl));
    xt::dump_npy("../plot_data/target.npy", xt::degrees(xt::col(trTarget, 0)));
    xt::dump_npy("../plot_data/estimation.npy", xt::degrees(xt::col(data.trEst[0], 0)));
    xt::dump_npy("../plot_data/target2.npy", xt::degrees(xt::col(trTarget, 1)));
    xt::dump_npy("../plot_data/estimation2.npy", xt::degrees(xt::col(data.trEst[0], 1)));
    xt::dump_npy("../plot_data/wdiff1.npy", wdiff1);
    xt::dump_npy("../plot_data/wdiff2.npy", wdiff2);
    xt::dump_npy("../plot_data/neuro1.npy", data.n1);
    xt::dump_npy("../plot_data/neuro2.npy", data.n2);
  }

  static double timeit(std::function<void(void)> foo, std::uint32_t count = 1)
  {
    std::vector<double> times(count);

    for(auto i = 0u; i < count; ++i) {
      auto begin = std::chrono::steady_clock::now();
      foo();
      auto end = std::chrono::steady_clock::now();
      times[i] = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() / 3305.;
    }

    return std::accumulate(times.begin(), times.end(), 0.) / count;
  }

  static std::unique_ptr<cxx_sdnn::SpikeDNNet> make_dnn(
    double alpha, std::uint32_t dim, std::unique_ptr<cxx_sdnn::AbstractActivation> act1,
    std::unique_ptr<cxx_sdnn::AbstractActivation> act2, std::unordered_map<std::string, xt::xarray<double>> kwargs)
  {
    return std::make_unique<cxx_sdnn::SpikeDNNet>(
      std::move(act1), std::move(act2), kwargs["W_1"], kwargs["W_2"], dim, kwargs["A"], kwargs["P"], kwargs["K_1"],
      kwargs["K_2"], alpha);
  }
};
