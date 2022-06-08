#include "IzhikevichActivation.hpp"

#include "debug_header.hpp"

#include "xtensor-blas/xlinalg.hpp"
#include "xtensor/xmasked_view.hpp"
#include "xtensor/xview.hpp"

using CxxSDNN::IzhikevichActivation;

IzhikevichActivation::IzhikevichActivation(
  double i_scale, double o_scale, double _izh_border, double a, double b, double c, double d, double e,
  std::vector<size_t> _shape) :
  input_scale{i_scale},
  output_scale{o_scale}, izh_border{_izh_border}, param_a{a}, param_b{b}, param_c{c}, param_d{d}, param_e{e}, shape{
                                                                                                                _shape}
{
  control = xt::eval(xt::ones<double>(shape) * param_b * param_e);
  state   = xt::eval(xt::ones<double>(shape) * param_e);
}

IzhikevichActivation::IzhikevichActivation(
  double i_scale, double o_scale, double _izh_border, double a, double b, double c, double d, double e) :
  IzhikevichActivation(i_scale, o_scale, _izh_border, a, b, c, d, e, {2})
{}

IzhikevichActivation::IzhikevichActivation(double i_scale, double o_scale) :
  IzhikevichActivation(i_scale, o_scale, 30, 2e-2, 0.2, -65, 8, -65, {2})
{}

IzhikevichActivation::IzhikevichActivation(double _izh_border) :
  IzhikevichActivation(80., 1 / 60., _izh_border, 2e-2, 0.2, -65, 8, -65)
{}

IzhikevichActivation::IzhikevichActivation(std::vector<size_t> _shape) :
  IzhikevichActivation(80., 1 / 60., 30, 2e-2, 0.2, -65, 8, -65, _shape)
{}

IzhikevichActivation::IzhikevichActivation() : IzhikevichActivation(80., 1 / 60., 30, 2e-2, 0.2, -65, 8, -65, {2})
{}

xt::xarray<double> IzhikevichActivation::operator()(xt::xarray<double> input, double step = .01)
{
  xt::xarray<double> vec_scale = xt::ones<double>(this->shape);
  xt::xarray<double> cur_input =
    xt::broadcast(input.reshape({this->shape[0] / 2, 1}) * this->input_scale, {this->shape[0] / 2, this->shape[1]});

  // Duplicate input to negate the second half
  cur_input = xt::vstack(xt::xtuple(cur_input, cur_input));

  // Enable the analog of python's a[start:] slice
  using namespace xt::placeholders;
  // Negate second half of input to take into account negative inputs
  auto half_of_input = xt::view(cur_input, xt::range(this->shape[0] / 2, _));
  half_of_input *= -1.;

  this->state += step / 2 * (.04 * this->state * this->state + 5. * this->state + 140. - this->control + cur_input);

  this->state += step / 2 * (.04 * this->state * this->state + 5. * this->state + 140. - this->control + cur_input);

  this->control += step * (this->param_a * (this->param_b * this->state - this->control));

  auto beyond_border    = this->state > this->izh_border;
  auto state_beyond     = xt::masked_view(this->state, beyond_border);
  auto control_by_state = xt::masked_view(this->control, beyond_border);

  state_beyond     = vec_scale * this->param_c;
  control_by_state = vec_scale * this->param_d;

  return this->state * this->output_scale;
}

xt::xarray<double> IzhikevichActivation::operator()(xt::xarray<double> input, double step) const
{
  xt::xarray<double> vec_scale = xt::ones<double>(this->shape);
  xt::xarray<double> cur_input =
    xt::broadcast(input.reshape({this->shape[0] / 2, 1}) * this->input_scale, {this->shape[0] / 2, this->shape[1]});

  xt::xarray<double> temp_state   = this->state;
  xt::xarray<double> temp_control = this->control;

  // Duplicate input to negate the second half
  cur_input = xt::vstack(xt::xtuple(cur_input, cur_input));

  // Enable the analog of python's a[start:] slice
  using namespace xt::placeholders;
  // Negate second half of input to take into account negative inputs
  auto half_of_input = xt::view(cur_input, xt::range(this->shape[0] / 2, _));
  half_of_input *= -1.;

  temp_state += step / 2 * (.04 * this->state * this->state + 5. * this->state + 140. - this->control + cur_input);

  temp_state += step / 2 * (.04 * this->state * this->state + 5. * this->state + 140. - this->control + cur_input);

  temp_control += step * (this->param_a * (this->param_b * this->state - this->control));

  auto beyond_border = temp_state > this->izh_border;
  auto state_beyond  = xt::masked_view(temp_state, beyond_border);
  // No need of changes in control, thus this function should not change neuron
  // state auto control_by_state = xt::masked_view(this->control,
  // beyond_border);

  state_beyond = vec_scale * this->param_c;
  // control_by_state = vec_scale * this->param_d;

  return temp_state * this->output_scale;
}

void IzhikevichActivation::set_type(NeuronType new_type)
{
  this->type = new_type;
}

const std::string IzhikevichActivation::whoami() const
{
  std::string out;

  out += "\tIzhikevich neuron pack\n\tShape: {";
  for(auto x : this->shape) {
    out += std::to_string(x) + ", ";
  }

  out += "}\n\tNeuron type: ";

  switch(this->type) {
  case NeuronType::Chattering :
    out += "Chattering";
    break;

  case NeuronType::RegularSpiking :
    out += "Regular Spiking";
    break;

  case NeuronType::Resonator :
    out += "Resonator";
    break;

  case NeuronType::LowThresholdSpiking :
    out += "Low Threshold Spiking";
    break;

  case NeuronType::ThalamoCortical63 :
    out += "Thalamo Cortical with -63 mV initial";
    break;

  case NeuronType::ThalamoCortical87 :
    out += "Thalamo Cortical with -87 mV initial";
    break;

  case NeuronType::IntrinsicallyBursting :
    out += "Intrinsically Bursting";
    break;
  case NeuronType::FastSpiking :
    out += "Fast Spiking";
    break;
  default :
    out += "Custom";
  }

  out += "\n\tParameters:\n";
  out += "\t\tIzhikevich border = " + std::to_string(this->izh_border) + "\n";
  out += "\t\ta = " + std::to_string(this->param_a) + "\n";
  out += "\t\tb = " + std::to_string(this->param_b) + "\n";
  out += "\t\tc = " + std::to_string(this->param_c) + "\n";
  out += "\t\td = " + std::to_string(this->param_d) + "\n";
  out += "\t\te = " + std::to_string(this->param_e) + "\n";
  out += "\t\tInput scale = " + std::to_string(this->input_scale) + "\n";
  out += "\t\tOutput scale = " + std::to_string(this->output_scale) + "\n";

  return out;
}

std::unique_ptr<CxxSDNN::IzhikevichActivation> make_izhikevich(
  double input_scale, double output_scale, std::vector<size_t> shape, IzhikevichActivation::NeuronType type)
{
  using Izhi = IzhikevichActivation;
  std::unique_ptr<Izhi> out;

  switch(type) {
  case Izhi::NeuronType::RegularSpiking :
    out = std::make_unique<Izhi>(input_scale, output_scale, 50, 0.02, 2., -65, 8, -65, shape);
    out->set_type(Izhi::NeuronType::RegularSpiking);
    break;
  case Izhi::NeuronType::IntrinsicallyBursting :
    out = std::make_unique<Izhi>(input_scale, output_scale, 50, 0.02, 2., -55, 4, -65, shape);
    out->set_type(Izhi::NeuronType::IntrinsicallyBursting);
    break;
  case Izhi::NeuronType::Chattering :
    out = std::make_unique<Izhi>(input_scale, output_scale, 50, 0.02, 2., -50, 2, -65, shape);
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
    out = std::make_unique<Izhi>(input_scale, output_scale, 40, 0.02, 0.25, -65, 0.05, -63, shape);
    out->set_type(Izhi::NeuronType::ThalamoCortical63);
    break;
  case Izhi::NeuronType::ThalamoCortical87 :
    out = std::make_unique<Izhi>(input_scale, output_scale, 50, 0.02, 0.25, -65, 0.05, -87, shape);
    out->set_type(Izhi::NeuronType::ThalamoCortical87);
    break;
  case Izhi::NeuronType::Resonator :
    out = std::make_unique<Izhi>(input_scale, output_scale, 30, 0.1, 0.26, -65, 2, -65, shape);
    out->set_type(Izhi::NeuronType::Resonator);
    break;
  }

  return out;
}
