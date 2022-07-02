#include "ActivationFunctions/IzhikevichActivation.hpp"

#include "Utility/debug_header.hpp"

#include "xtensor-blas/xlinalg.hpp"
#include "xtensor/xmasked_view.hpp"
#include "xtensor/xview.hpp"

using cxx_sdnn::IzhikevichActivation;

IzhikevichActivation::IzhikevichActivation(
  double iScale, double oScale, double izhBorder, double a, double b, double c, double d, double e,
  std::vector<size_t> shape) :
  Super(shape),
  inputScale{iScale}, outputScale{oScale}, izhBorder{izhBorder}, paramA{a}, paramB{b}, paramC{c}, paramD{d}, paramE{e}
{
  control = xt::eval(xt::ones<double>(shape) * paramB * paramE);
  state   = xt::eval(xt::ones<double>(shape) * paramE);
}

IzhikevichActivation::IzhikevichActivation(
  double iScale, double oScale, double izhBorder, double a, double b, double c, double d, double e) :
  IzhikevichActivation(iScale, oScale, izhBorder, a, b, c, d, e, {2})
{}

IzhikevichActivation::IzhikevichActivation(double iScale, double oScale) :
  IzhikevichActivation(iScale, oScale, 30, 2e-2, 0.2, -65, 8, -65, {2})
{}

IzhikevichActivation::IzhikevichActivation(double izhBorder) :
  IzhikevichActivation(80., 1 / 60., izhBorder, 2e-2, 0.2, -65, 8, -65)
{}

IzhikevichActivation::IzhikevichActivation(std::vector<size_t> shape) :
  IzhikevichActivation(80., 1 / 60., 30, 2e-2, 0.2, -65, 8, -65, shape)
{}

IzhikevichActivation::IzhikevichActivation() : IzhikevichActivation(80., 1 / 60., 30, 2e-2, 0.2, -65, 8, -65, {2})
{}

xt::xarray<double> IzhikevichActivation::operator()(xt::xarray<double> input, double step = .01)
{
  xt::xarray<double> vecScale = xt::ones<double>(this->shape);
  xt::xarray<double> curInput =
    xt::broadcast(input.reshape({this->shape[0] / 2, 1}) * this->inputScale, {this->shape[0] / 2, this->shape[1]});

  // Duplicate input to negate the second half
  curInput = xt::vstack(xt::xtuple(curInput, curInput));

  // Enable the analog of python's a[start:] slice
  using namespace xt::placeholders;
  // Negate second half of input to take into account negative inputs
  auto halfOfInput = xt::view(curInput, xt::range(this->shape[0] / 2, _));
  halfOfInput *= -1.;

  this->state += step / 2 * (.04 * this->state * this->state + 5. * this->state + 140. - this->control + curInput);

  this->state += step / 2 * (.04 * this->state * this->state + 5. * this->state + 140. - this->control + curInput);

  this->control += step * (this->paramA * (this->paramB * this->state - this->control));

  auto beyondBorder   = this->state > this->izhBorder;
  auto stateBeyond    = xt::masked_view(this->state, beyondBorder);
  auto controlByState = xt::masked_view(this->control, beyondBorder);

  stateBeyond    = vecScale * this->paramC;
  controlByState = vecScale * this->paramD;

  return this->state * this->outputScale;
}

xt::xarray<double> IzhikevichActivation::operator()(xt::xarray<double> input, double step) const
{
  xt::xarray<double> vecScale = xt::ones<double>(this->shape);
  xt::xarray<double> curInput =
    xt::broadcast(input.reshape({this->shape[0] / 2, 1}) * this->inputScale, {this->shape[0] / 2, this->shape[1]});

  xt::xarray<double> tempState   = this->state;
  xt::xarray<double> tempControl = this->control;

  // Duplicate input to negate the second half
  curInput = xt::vstack(xt::xtuple(curInput, curInput));

  // Enable the analog of python's a[start:] slice
  using namespace xt::placeholders;
  // Negate second half of input to take into account negative inputs
  auto halfOfInput = xt::view(curInput, xt::range(this->shape[0] / 2, _));
  halfOfInput *= -1.;

  tempState += step / 2 * (.04 * this->state * this->state + 5. * this->state + 140. - this->control + curInput);

  tempState += step / 2 * (.04 * this->state * this->state + 5. * this->state + 140. - this->control + curInput);

  tempControl += step * (this->paramA * (this->paramB * this->state - this->control));

  auto beyondBorder = tempState > this->izhBorder;
  auto stateBeyond  = xt::masked_view(tempState, beyondBorder);
  // No need in changes of control, thus this function should not change neuron
  // state
  // auto control_by_state = xt::masked_view(this->control,
  // beyond_border);

  stateBeyond = vecScale * this->paramC;
  // control_by_state = vec_scale * this->param_d;

  return tempState * this->outputScale;
}

void IzhikevichActivation::set_type(NeuronType newType)
{
  this->type = newType;
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
  case NeuronType::CHATTERING :
    out += "Chattering";
    break;

  case NeuronType::REGULAR_SPIKING :
    out += "Regular Spiking";
    break;

  case NeuronType::RESONATOR :
    out += "Resonator";
    break;

  case NeuronType::LOW_THRESHOLD_SPIKING :
    out += "Low Threshold Spiking";
    break;

  case NeuronType::THALAMO_CORTICAL63 :
    out += "Thalamo Cortical with -63 mV initial";
    break;

  case NeuronType::THALAMO_CORTICAL87 :
    out += "Thalamo Cortical with -87 mV initial";
    break;

  case NeuronType::INTRINSICALLY_BURSTING :
    out += "Intrinsically Bursting";
    break;
  case NeuronType::FAST_SPIKING :
    out += "Fast Spiking";
    break;
  default :
    out += "Custom";
  }

  out += "\n\tParameters:\n";
  out += "\t\tIzhikevich border = " + std::to_string(this->izhBorder) + "\n";
  out += "\t\ta = " + std::to_string(this->paramA) + "\n";
  out += "\t\tb = " + std::to_string(this->paramB) + "\n";
  out += "\t\tc = " + std::to_string(this->paramC) + "\n";
  out += "\t\td = " + std::to_string(this->paramD) + "\n";
  out += "\t\te = " + std::to_string(this->paramE) + "\n";
  out += "\t\tInput scale = " + std::to_string(this->inputScale) + "\n";
  out += "\t\tOutput scale = " + std::to_string(this->outputScale) + "\n";

  return out;
}

std::unique_ptr<cxx_sdnn::IzhikevichActivation>
make_izhikevich(double inputScale, double outputScale, std::vector<size_t> shape, IzhikevichActivation::NeuronType type)
{
  using Izhi = IzhikevichActivation;
  std::unique_ptr<Izhi> out;

  switch(type) {
  case Izhi::NeuronType::REGULAR_SPIKING :
    out = std::make_unique<Izhi>(inputScale, outputScale, 50, 0.02, 2., -65, 8, -65, shape);
    out->set_type(Izhi::NeuronType::REGULAR_SPIKING);
    break;
  case Izhi::NeuronType::INTRINSICALLY_BURSTING :
    out = std::make_unique<Izhi>(inputScale, outputScale, 50, 0.02, 2., -55, 4, -65, shape);
    out->set_type(Izhi::NeuronType::INTRINSICALLY_BURSTING);
    break;
  case Izhi::NeuronType::CHATTERING :
    out = std::make_unique<Izhi>(inputScale, outputScale, 50, 0.02, 2., -50, 2, -65, shape);
    out->set_type(Izhi::NeuronType::CHATTERING);
    break;
  case Izhi::NeuronType::FAST_SPIKING :
    out = std::make_unique<Izhi>(inputScale, outputScale, 50, 0.1, 0.2, -65, 2, -65, shape);
    out->set_type(Izhi::NeuronType::FAST_SPIKING);
    break;
  case Izhi::NeuronType::LOW_THRESHOLD_SPIKING :
    out = std::make_unique<Izhi>(inputScale, outputScale, 30, 0.02, 0.25, -65, 2, -65, shape);
    out->set_type(Izhi::NeuronType::LOW_THRESHOLD_SPIKING);
    break;
  case Izhi::NeuronType::THALAMO_CORTICAL63 :
    out = std::make_unique<Izhi>(inputScale, outputScale, 40, 0.02, 0.25, -65, 0.05, -63, shape);
    out->set_type(Izhi::NeuronType::THALAMO_CORTICAL63);
    break;
  case Izhi::NeuronType::THALAMO_CORTICAL87 :
    out = std::make_unique<Izhi>(inputScale, outputScale, 50, 0.02, 0.25, -65, 0.05, -87, shape);
    out->set_type(Izhi::NeuronType::THALAMO_CORTICAL87);
    break;
  case Izhi::NeuronType::RESONATOR :
    out = std::make_unique<Izhi>(inputScale, outputScale, 30, 0.1, 0.26, -65, 2, -65, shape);
    out->set_type(Izhi::NeuronType::RESONATOR);
    break;
  }

  return out;
}
