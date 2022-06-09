#pragma once

#include "xtensor/xarray.hpp"
#include "xtensor/xio.hpp"

#include <memory>
#include <string>
#include <unordered_map>

namespace cxx_sdnn
{
  class AbstractActivation;

  class SpikeDNNet {
  private:
    std::unique_ptr<AbstractActivation> afunc1;
    std::unique_ptr<AbstractActivation> afunc2;
    xt::xarray<double> matA;
    xt::xarray<double> matP;
    xt::xarray<double> matK1;
    xt::xarray<double> matK2;
    xt::xarray<double> initMatW1;
    xt::xarray<double> initMatW2;
    size_t matDim;

    xt::xarray<double> matW1;
    xt::xarray<double> matW2;
    xt::xarray<double> arrayHistW1;
    xt::xarray<double> arrayHistW2;
    xt::xarray<double> smoothedW1;
    xt::xarray<double> smoothedW2;
    xt::xarray<double> neuron1Hist;
    xt::xarray<double> neuron2Hist;

  public:
    explicit SpikeDNNet(
      std::unique_ptr<AbstractActivation> actFunc1, std::unique_ptr<AbstractActivation> actFunc2,
      xt::xarray<double> matW1, xt::xarray<double> matW2, size_t dim = 2,
      xt::xarray<double> matA   = 20. * xt::diag(xt::xarray<double>({-1., -2.})),
      xt::xarray<double> matP   = 1575.9 * xt::diag(xt::xarray<double>({60., 40.})),
      xt::xarray<double> matK1 = .15 * xt::diag(xt::xarray<double>({10., 1.})),
      xt::xarray<double> matK2 = .15 * xt::diag(xt::xarray<double>({1., 1.})));

    SpikeDNNet(const SpikeDNNet& other) noexcept;

    SpikeDNNet& operator=(const SpikeDNNet& other) noexcept;

    static xt::xarray<double> moving_average(xt::xarray<double> x, std::uint32_t w = 2);

    xt::xarray<double> smooth(xt::xarray<double> x, std::uint32_t w = 2);

    xt::xarray<double> fit(
      xt::xarray<double> vecX, xt::xarray<double> vecU, double step = .01, std::uint32_t nEpochs = 3,
      std::uint32_t kPoints = 2);

    xt::xarray<double> predict(xt::xarray<double> initState, xt::xarray<double> vecU, double step = .01);

    xt::xarray<double> get_weights(std::uint8_t idx) const;

    xt::xarray<double> get_neurons_history(std::uint8_t idx) const;

    const xt::xarray<double>& get_A() const;
    const xt::xarray<double>& get_P() const;
    const xt::xarray<double>& get_K1() const;
    const xt::xarray<double>& get_K2() const;
    const xt::xarray<double>& get_W10() const;
    const xt::xarray<double>& get_W20() const;
    const std::string get_afunc_descr(size_t idx) const;

    friend std::ostream& operator<<(std::ostream& out, const SpikeDNNet& dnn)
    {
      out << "\tSDNN model.\n";
      out << "\n=================================\n";
      out << "\tConfiguration: \n";
      out << "\t\tA =\n" << dnn.get_A() << '\n';
      out << "\t\tP =\n" << dnn.get_P() << '\n';
      out << "\t\tW10 =\n" << dnn.get_W10() << '\n';
      out << "\t\tW20 =\n" << dnn.get_W20() << '\n';
      out << "\t\tK1 =\n" << dnn.get_K1() << '\n';
      out << "\t\tK2 =\n" << dnn.get_K2() << '\n';
      out << "\n=================================\n\n";
      out << "\tFirst activation component: \n" << dnn.get_afunc_descr(0) << '\n';
      out << "\n=================================\n\n";
      out << "\tSecond activation component: \n" << dnn.get_afunc_descr(1) << '\n';
      out << "\t\t\n=================================\n\n\n";

      return out;
    }
  };

  // std::unique_ptr<SpikeDNNet> make_dnn(
  //   std::uint32_t dim, std::unique_ptr<AbstractActivation> act_1, std::unique_ptr<AbstractActivation> act_2,
  //   std::unordered_map<std::string, xt::xarray<double>> kwargs);
}; // namespace CxxSDNN
