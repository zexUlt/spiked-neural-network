#pragma once

#include "AbstractActivation.hpp"

namespace cxx_sdnn
{
  class SigmoidActivation : public AbstractActivation {
  private:
    typedef AbstractActivation Super;

    double paramA;
    double paramB;
    double paramC;
    double paramD;

  public:
    explicit SigmoidActivation(
      std::vector<size_t> shape, double a = 1., double b = 1., double c = .02, double d = -.02);
    xt::xarray<double> operator()(xt::xarray<double> input, double step) override;
    xt::xarray<double> operator()(xt::xarray<double> input, double step) const override;
    const std::string whoami() const override;
  };

}; // namespace CxxSDNN