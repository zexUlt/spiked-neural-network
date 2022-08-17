#pragma once

#include "AbstractActivation.hpp"

namespace cxx_sdnn
{
  class IzhikevichActivation : public AbstractActivation {
  public:
    enum class NeuronType
    {
      REGULAR_SPIKING,
      INTRINSICALLY_BURSTING,
      CHATTERING,
      FAST_SPIKING,
      LOW_THRESHOLD_SPIKING,
      THALAMO_CORTICAL63,
      THALAMO_CORTICAL87,
      RESONATOR,
      CUSTOM
    };

    explicit IzhikevichActivation(
      double iScale, double oScale, double izhBorder, double a, double b, double c, double d, double e,
      std::vector<size_t> shape);

    explicit IzhikevichActivation(
      double iScale, double oScale, double izhBorder, double a, double b, double c, double d, double e);

    explicit IzhikevichActivation(double iScale, double oScale);

    explicit IzhikevichActivation(double izhBorder);

    explicit IzhikevichActivation(std::vector<size_t> shape);

    explicit IzhikevichActivation();

    xt::xarray<double> operator()(xt::xarray<double> input) override;
    xt::xarray<double> operator()(xt::xarray<double> input) const override;
    const std::string whoami() const override;
    void set_type(NeuronType newType);
    void set_integration_step(double new_step);

  private:
    typedef AbstractActivation Super;

    double inputScale;
    double outputScale;
    double izhBorder;
    double paramA;
    double paramB;
    double paramC;
    double paramD;
    double paramE;
    double step;
    xt::xarray<double> control;
    xt::xarray<double> state;
    NeuronType type = NeuronType::CUSTOM;
  };

  std::unique_ptr<cxx_sdnn::IzhikevichActivation> make_izhikevich(
    double inputScale, double outputScale, std::vector<size_t> shape, IzhikevichActivation::NeuronType type);

}; // namespace cxx_sdnn
