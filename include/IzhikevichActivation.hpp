#pragma once

#include "AbstractActivation.hpp"

namespace CxxSDNN
{
  class IzhikevichActivation : public AbstractActivation {
  public:
    enum class NeuronType
    {
      RegularSpiking,
      IntrinsicallyBursting,
      Chattering,
      FastSpiking,
      LowThresholdSpiking,
      ThalamoCortical63,
      ThalamoCortical87,
      Resonator,
      Custom
    };

    explicit IzhikevichActivation(
      double i_scale, double o_scale, double _izh_border, double a, double b, double c, double d, double e,
      std::vector<size_t> _shape);

    explicit IzhikevichActivation(
      double i_scale, double o_scale, double _izh_border, double a, double b, double c, double d, double e);

    explicit IzhikevichActivation(double i_scale, double o_scale);

    explicit IzhikevichActivation(double _izh_border);

    explicit IzhikevichActivation(std::vector<size_t> _shape);

    explicit IzhikevichActivation();

    xt::xarray<double> operator()(xt::xarray<double> input, double step) override;
    const std::string whoami() const override;
    void set_type(NeuronType new_type);

  private:
    double input_scale;
    double output_scale;
    double izh_border;
    double param_a;
    double param_b;
    double param_c;
    double param_d;
    double param_e;
    std::vector<size_t> shape;
    xt::xarray<double> control;
    xt::xarray<double> state;
    NeuronType type = NeuronType::Custom;
  };

}; // namespace CxxSDNN
