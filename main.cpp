#include "IzhikevichActivation.hpp"
#include "SigmoidActivation.hpp"
#include "SpikeDNNet.hpp"
#include "Utility.hpp"
#include "debug_header.hpp"
// #include "Optimization.hpp"
#include "Integrate.hpp"
#include "precompiled.hpp"

xt::xarray<double> f(double t, xt::xarray<double> x)
{
  return xt::exp(xt::xarray<double>{t});
}

int main(int argc, char** argv)
{
  // auto integrator = cxx_sdnn::RungeKutta4NonAdapt(f);
  // xt::xarray<double> data(std::vector<size_t>({1, 1000})), exponent(std::vector<size_t>{1, 1000});
  
  // data[0] = 1.; // Initial condition
  
  // integrator.integrate(data, 0., 10, 1000);

  // double h = (10.) / 1000;
  // double t = 0.;
  
  // for(auto i = 0; i < 1000; ++i, t += h){
  //   exponent[i] = f(t, {})();
  // }

  // xt::dump_npy("../plot_data/integration_test_source.npy", exponent);
  // xt::dump_npy("../plot_data/integration_test.npy", data);
  
  auto folds = cxx_sdnn::UtilityFunctionLibrary::prepare_dataset("../train_data/");

  // const std::uint32_t width    = 4394u;
  // const std::int32_t split     = 3305;

  const std::uint32_t N_EPOCHS  = 1u;
  const std::uint32_t K_POINTS  = 3u;
  const double ALPHA            = 0.;
  const double INTEGRATION_STEP = 0.0001;

  std::uint32_t targetDim = folds["tr"].first.shape(1);
  std::uint32_t controlDim = folds["tr"].second.shape(1);

  std::unordered_map<std::string, xt::xarray<double>> modelParams{
    {"W_1", 1000. * xt::ones<double>({targetDim, targetDim})},
    {"W_2", 1000. * xt::ones<double>({targetDim, targetDim})},
    {"A", 5. * xt::diag(xt::xarray<double>{-25., -25., -18., -19.})},
    {"P", 500. * xt::xarray<double>{{18., 1., 6., 0.2},
                                    {1., 25., 5., 0.4},
                                    {6., 5., 132., 0.8},
                                    {0.2, 0.4, 0.8, 252.}}}, // [18 1 6 0.2; 1 25 5 0.4; 6 5 132 0.8; 0.2 0.4 0.8 252]*500
    {"K_1", 1. * xt::diag(xt::xarray<double>{-0.1724, -0.1724, -0.1724, -0.1724})},
    {"K_2", 1. * xt::diag(xt::xarray<double>{-2., -2., -2., -2.})}};

  // auto izh_act_1 = cxx_sdnn::make_izhikevich(50, 1/40., {2*1, 1}, cxx_sdnn::IzhikevichActivation::NeuronType::RESONATOR);
  // auto izh_act_2 = cxx_sdnn::make_izhikevich(55, 1/40., {2*1, 1}, cxx_sdnn::IzhikevichActivation::NeuronType::THALAMO_CORTICAL63);

  auto act1 = std::make_unique<cxx_sdnn::SigmoidActivation>(
    /*shape=*/std::vector<size_t>{targetDim, 1},
    /*a =*/2.,
    /*b =*/1.,
    /*c =*/1,
    /*d =*/-0.2,
    /*e =*/0.5);
  auto act2 = std::make_unique<cxx_sdnn::SigmoidActivation>(
    /*shape=*/std::vector<size_t>{targetDim, controlDim},
    /*a =*/2.,
    /*b =*/1.,
    /*c =*/1,
    /*d =*/-0.2,
    /*e =*/0.5);
    
  auto model =
    cxx_sdnn::UtilityFunctionLibrary::make_dnn(ALPHA, targetDim, std::move(act1), std::move(act2), modelParams);

  auto res = cxx_sdnn::UtilityFunctionLibrary::dnn_validate(std::move(model), folds, N_EPOCHS, K_POINTS, INTEGRATION_STEP);

  std::cout << res;

  cxx_sdnn::UtilityFunctionLibrary::dump_data("../plot_data/", folds["tr"].first, folds["tr"].second, res);

  return 0;
}