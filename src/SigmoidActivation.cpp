#include "SigmoidActivation.hpp"

using CxxSDNN::SigmoidActivation;

SigmoidActivation::SigmoidActivation(double a, double b, double c, double d) :
  param_a{a}, param_b{b}, param_c{c}, param_d{d}
{}

xt::xarray<double>
SigmoidActivation::operator()(xt::xarray<double> input, double step = -1.)
{
  return this->param_a /
         (this->param_b + this->param_c * xt::exp(this->param_d * input));
}

const std::string SigmoidActivation::whoami() const
{
  std::string out;

  out += "Sigmoid Activation pack.\n";
  out += "Parameters:\n";
  out += "a = " + std::to_string(this->param_a) + "\n";
  out += "b = " + std::to_string(this->param_b) + "\n";
  out += "c = " + std::to_string(this->param_c) + "\n";
  out += "d = " + std::to_string(this->param_d) + "\n";

  return out;
}