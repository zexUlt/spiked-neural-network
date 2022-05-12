#pragma once

#include "IzhikevichActivation.hpp"
#include "SpikeDNNet.hpp"
#include "debug_header.hpp"

#include "xtensor-blas/xlinalg.hpp"
#include "xtensor/xnpy.hpp"
#include "xtensor/xview.hpp"

#include <chrono>
#include <map>
#include <string>

class UtilityFunctionLibrary {
public:
  template<typename dtype>
  using vl_tr_map = std::map<std::string, std::pair<xt::xarray<dtype>, xt::xarray<dtype>>>;
  using Izhi      = CxxSDNN::IzhikevichActivation;

  struct ValidationResults
  {
    std::pair<double, double> mse_res;
    std::pair<double, double> mae_res;
    std::vector<xt::xarray<double>> tr_est;
    std::vector<xt::xarray<double>> vl_res;
    xt::xarray<double> W_1;
    xt::xarray<double> W_2;
    xt::xarray<double> N_1;
    xt::xarray<double> N_2;

    friend std::ostream& operator<<(std::ostream& out, UtilityFunctionLibrary::ValidationResults res)
    {
      out << "MSE: [Train: " << res.mse_res.first << ", Test: " << res.mse_res.second << "]\n";
      out << "MAE: [Train: " << res.mae_res.first << ", Test: " << res.mae_res.second << "]\n";

      return out;
    }
  };

  static xt::xarray<double> convolveValid(const xt::xarray<double>& f, const xt::xarray<double>& g)
  {
    const auto nf = f.size();
    const auto ng = g.size();

    const auto& min_v = (nf < ng) ? f : g;
    const auto& max_v = (nf < ng) ? g : f;
    const auto n      = std::max(nf, ng) - std::min(nf, ng) + 1;

    xt::xarray<double> out = xt::zeros<double>({n});

    for(auto i(0u); i < n; ++i) {
      for(int j(min_v.size() - 1), k(i); j >= 0; --j, ++k) {
        out.at(i) += min_v[j] * max_v[k];
      }
    }

    return out;
  }

  static ValidationResults
  dnn_validate(CxxSDNN::SpikeDNNet& dnn, vl_tr_map<double> folds, std::uint16_t n_epochs, std::uint16_t k_points)
  {
    ValidationResults results;

    std::cout << dnn;

    auto tr_target  = folds["tr"].first;
    auto tr_control = folds["tr"].second;

    auto vl_target  = folds["vl"].first;
    auto vl_control = folds["vl"].second;

    auto target_est = dnn.fit(tr_target, tr_control, 0.01, n_epochs, k_points);
    auto vl_pred    = dnn.predict(xt::view(tr_target, -1, 0), vl_control);

    xt::xarray<double> tr_col     = xt::col(tr_target, 0);
    xt::xarray<double> target_col = xt::col(target_est, 0);
    xt::xarray<double> vl_col     = xt::col(vl_target, 0);
    xt::xarray<double> pred_col   = xt::col(vl_pred, 0);
    results.mse_res               = std::make_pair(
      UtilityFunctionLibrary::mean_squared_error<double>(tr_col, target_col),
      UtilityFunctionLibrary::mean_squared_error<double>(vl_col, pred_col));

    results.mae_res = std::make_pair(
      UtilityFunctionLibrary::mean_absolute_error<double>(tr_col, target_col),
      UtilityFunctionLibrary::mean_absolute_error<double>(vl_col, pred_col));

    results.tr_est.emplace_back(target_est);
    results.vl_res.emplace_back(vl_pred);
    results.W_1 = dnn.get_weights(0);
    results.W_2 = dnn.get_weights(1);
    results.N_1 = dnn.get_neurons_history(0);
    results.N_2 = dnn.get_neurons_history(1);

    return results;
  }

  template<typename dtype>
  static double mean_squared_error(xt::xarray<dtype> y_true, xt::xarray<dtype> y_pred)
  {
    xt::xarray<dtype> sq = xt::square(y_true - y_pred);
    auto output_errors   = xt::average(sq);

    return xt::average(output_errors)();
  }

  template<typename dtype>
  static double mean_absolute_error(xt::xarray<dtype> y_true, xt::xarray<dtype> y_pred)
  {
    xt::xarray<dtype> absol = xt::abs(y_pred - y_true);
    auto output_errors      = xt::average(absol);

    return xt::average(output_errors)();
  }

  static void dumpData(xt::xarray<double> tr_target, xt::xarray<double> tr_control, ValidationResults data)
  {
    auto error  = xt::abs(xt::col(tr_target, 0) - xt::col(data.tr_est[0], 0));
    auto wdiff1 = xt::diff(xt::view(data.W_1, xt::all(), xt::all(), 0), 1, 0);
    auto wdiff2 = xt::diff(xt::view(data.W_2, xt::all(), xt::all(), 0), 1, 0);

    xt::dump_npy("../plot_data/error.npy", xt::degrees(error));
    xt::dump_npy("../plot_data/control.npy", xt::degrees(tr_control));
    xt::dump_npy("../plot_data/target.npy", xt::degrees(xt::col(tr_target, 0)));
    xt::dump_npy("../plot_data/estimation.npy", xt::degrees(xt::col(data.tr_est[0], 0)));
    xt::dump_npy("../plot_data/target2.npy", xt::degrees(xt::col(tr_target, 1)));
    xt::dump_npy("../plot_data/estimation2.npy", xt::degrees(xt::col(data.tr_est[0], 1)));
    xt::dump_npy("../plot_data/wdiff1.npy", wdiff1);
    xt::dump_npy("../plot_data/wdiff2.npy", wdiff2);
    xt::dump_npy("../plot_data/neuro1.npy", data.N_1);
    xt::dump_npy("../plot_data/neuro2.npy", data.N_2);
  }

  static std::unique_ptr<CxxSDNN::IzhikevichActivation>
  make_izhikevich(double input_scale, double output_scale, std::vector<size_t> shape, Izhi::NeuronType type)
  {
    std::unique_ptr<Izhi> out;

    switch(type) {
    case Izhi::NeuronType::RegularSpiking :
      out = std::make_unique<Izhi>(input_scale, output_scale, 50, 0.02, 0.2, -65, 8, -65, shape);
      out->set_type(Izhi::NeuronType::RegularSpiking);
      break;
    case Izhi::NeuronType::IntrinsicallyBursting :
      out = std::make_unique<Izhi>(input_scale, output_scale, 50, 0.02, 0.2, -55, 4, -65, shape);
      out->set_type(Izhi::NeuronType::IntrinsicallyBursting);
      break;
    case Izhi::NeuronType::Chattering :
      out = std::make_unique<Izhi>(input_scale, output_scale, 50, 0.02, 0.2, -50, 2, -65, shape);
      out->set_type(Izhi::NeuronType::Chattering);
      break;
    case Izhi::NeuronType::FastSpiking :
      out = std::make_unique<Izhi>(input_scale, output_scale, 50, 0.1, 0.2, -65, 2, -65, shape);
      out->set_type(Izhi::NeuronType::FastSpiking);
      break;
    case Izhi::NeuronType::LowThresholdSpiking :
      out = std::make_unique<Izhi>(input_scale, output_scale, 30, 0.02, 0.25, -65, 2, -65, shape);
      out->set_type(Izhi::NeuronType::LowThresholdSpiking);
      break;
    case Izhi::NeuronType::ThalamoCortical63 :
      out = std::make_unique<Izhi>(input_scale, output_scale, 50, 0.02, 0.25, -65, 0.05, -63, shape);
      out->set_type(Izhi::NeuronType::ThalamoCortical63);
      break;
    case Izhi::NeuronType::ThalamoCortical87 :
      out = std::make_unique<Izhi>(input_scale, output_scale, 50, 0.02, 0.25, -65, 0.05, -87, shape);
      out->set_type(Izhi::NeuronType::ThalamoCortical87);
      break;
    case Izhi::NeuronType::Resonator :
      out = std::make_unique<Izhi>(input_scale, output_scale, 50, 0.1, 0.26, -65, 2, -65, shape);
      out->set_type(Izhi::NeuronType::Resonator);
      break;
    }

    return out;
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
};
