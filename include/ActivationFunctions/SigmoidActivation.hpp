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
    double paramE;

  public:
    explicit SigmoidActivation(
      std::vector<size_t> shape, double a = 1., double b = 1., double c = .02, double d = -.02, double e = -1.);
    xt::xarray<double> operator()(xt::xarray<double> input) override;
    xt::xarray<double> operator()(xt::xarray<double> input) const override;
    const std::string whoami() const override;
  };

}; // namespace cxx_sdnn