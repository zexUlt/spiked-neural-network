#include "SigmoidActivation.hpp"

using cxx_sdnn::SigmoidActivation;

SigmoidActivation::SigmoidActivation(std::vector<size_t> shape, double a, double b, double c, double d, double e) :
  Super(shape), paramA{a}, paramB{b}, paramC{c}, paramD{d}, paramE{e}
{}

xt::xarray<double> SigmoidActivation::operator()(xt::xarray<double> input, double step = -1.)
{
  return this->paramA /
         (this->paramB +
          this->paramC * xt::exp(this->paramD * (xt::broadcast(input.reshape({this->shape[0], 1}), this->shape) - 4.))) + this->paramE;
}

xt::xarray<double> SigmoidActivation::operator()(xt::xarray<double> input, double step = -1.) const
{
  return this->paramA /
         (this->paramB +
          this->paramC * xt::exp(this->paramD * (xt::broadcast(input.reshape({this->shape[0], 1}), this->shape) - 4.))) + this->paramE;
}

const std::string SigmoidActivation::whoami() const
{
  std::string out;

  out += "Sigmoid Activation pack.\n";
  out += "Parameters:\n";
  out += "a = " + std::to_string(this->paramA) + "\n";
  out += "b = " + std::to_string(this->paramB) + "\n";
  out += "c = " + std::to_string(this->paramC) + "\n";
  out += "d = " + std::to_string(this->paramD) + "\n";
  out += "e = " + std::to_string(this->paramE) + "\n";

  return out;
}