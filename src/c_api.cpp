#include "c_api.hpp"

#include "Optimization.hpp"
#include "SpikeDNNet.hpp"

extern "C"
{
  double min_functional(double a, double p, double k1, double k2, double alpha, const char* neuronType)
  {
    return cxx_sdnn::optimization::run_minimize(a, p, k1, k2, alpha, neuronType);
  }
}