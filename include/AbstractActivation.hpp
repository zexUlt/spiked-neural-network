#pragma once

#include <vector>
#include <xtensor/xarray.hpp>

namespace cxx_sdnn
{
  class AbstractActivation {
  public:
    explicit AbstractActivation(std::vector<size_t> shape) : shape{shape}
    {}

    virtual xt::xarray<double> operator()(xt::xarray<double> input, double step = 0)       = 0;
    virtual xt::xarray<double> operator()(xt::xarray<double> input, double step = 0) const = 0;
    virtual const std::string whoami() const                                               = 0;
    virtual ~AbstractActivation(){};

  protected:
    std::vector<size_t> shape;
  };

}; // namespace cxx_sdnn
