#pragma once

#include "precompiled.hpp"

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
    double alpha;

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
      xt::xarray<double> matA  = 20. * xt::diag(xt::xarray<double>({-1., -2.})),
      xt::xarray<double> matP  = 1575.9 * xt::diag(xt::xarray<double>({60., 40.})),
      xt::xarray<double> matK1 = .15 * xt::diag(xt::xarray<double>({10., 1.})),
      xt::xarray<double> matK2 = .15 * xt::diag(xt::xarray<double>({1., 1.})),
      double alpha = 1.);

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
      out << "SDNN model.\n\n";
      out << "=================================\n";
      out << "Configuration:\n\n";
      out << "Alpha = " << dnn.alpha << '\n';
      out << "A:\n" << dnn.get_A() << "\n\n";
      out << "P:\n" << dnn.get_P() << "\n\n";
      out << "W10:\n" << dnn.get_W10() << "\n\n";
      out << "W20:\n" << dnn.get_W20() << "\n\n";
      out << "K1:\n" << dnn.get_K1() << "\n\n";
      out << "K2:\n" << dnn.get_K2() << "\n\n";
      out << "=================================\n\n";
      out << "First activation component:\n\n" << dnn.get_afunc_descr(0) << "\n\n";
      out << "=================================\n\n";
      out << "Second activation component:\n\n" << dnn.get_afunc_descr(1) << '\n';
      out << "=================================\n\n\n";

      return out;
    }
  };
}; // namespace cxx_sdnn
