#pragma once

#include "AbstractActivation.hpp"

namespace CxxSDNN
{
  class SigmoidActivation : public AbstractActivation {
  private:
    double param_a;
    double param_b;
    double param_c;
    double param_d;

  public:
    explicit SigmoidActivation(double a = 1., double b = 1., double c = .02, double d = -.02);
    xt::xarray<double> operator()(xt::xarray<double> input, double step) override;
    const std::string whoami() const override;
  };

}; // namespace CxxSDNN