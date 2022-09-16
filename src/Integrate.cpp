#include "Integrate.hpp"
#include "debug_header.hpp"

using namespace cxx_sdnn;

constexpr double RungeKutta5Adapt::a21;
constexpr double RungeKutta5Adapt::a31;
constexpr double RungeKutta5Adapt::a32;
constexpr double RungeKutta5Adapt::a41;
constexpr double RungeKutta5Adapt::a42;
constexpr double RungeKutta5Adapt::a43;
constexpr double RungeKutta5Adapt::a51;
constexpr double RungeKutta5Adapt::a52;
constexpr double RungeKutta5Adapt::a53;
constexpr double RungeKutta5Adapt::a54;
constexpr double RungeKutta5Adapt::a61;
constexpr double RungeKutta5Adapt::a62;
constexpr double RungeKutta5Adapt::a64;
constexpr double RungeKutta5Adapt::a63;
constexpr double RungeKutta5Adapt::a65;
constexpr double RungeKutta5Adapt::a71;
constexpr double RungeKutta5Adapt::a72;
constexpr double RungeKutta5Adapt::a73;
constexpr double RungeKutta5Adapt::a74;
constexpr double RungeKutta5Adapt::a75;
constexpr double RungeKutta5Adapt::a76;

constexpr double RungeKutta5Adapt::c2;
constexpr double RungeKutta5Adapt::c3;
constexpr double RungeKutta5Adapt::c4;
constexpr double RungeKutta5Adapt::c5;
constexpr double RungeKutta5Adapt::c6;
constexpr double RungeKutta5Adapt::c7;

constexpr double RungeKutta5Adapt::b1;
constexpr double RungeKutta5Adapt::b2;
constexpr double RungeKutta5Adapt::b3;
constexpr double RungeKutta5Adapt::b4;
constexpr double RungeKutta5Adapt::b5;
constexpr double RungeKutta5Adapt::b6;
constexpr double RungeKutta5Adapt::b7;


RungeKutta5Adapt::RungeKutta5Adapt() : stepNumber{0}
{}

double RungeKutta5Adapt::step(const xt::xarray<double>& from, double t, double h, xt::xarray<double>& out)
{
  // k1 = f(t, x)
  out = from + h * RungeKutta5Adapt::a21 * this->k1;
  // k2 = f(t + c2 * h, out)
  out = from + h * (RungeKutta5Adapt::a31 * this->k1 + RungeKutta5Adapt::a32 * this->k2);
  // k3 = f(t + c3 * h, out)
  out = from + h * (RungeKutta5Adapt::a41 * this->k1 + RungeKutta5Adapt::a42 * this->k3 + RungeKutta5Adapt::a43 * this->k3);
  // k4 = f(t + c4 * h, out)
  out = from + h * (RungeKutta5Adapt::a51 * this->k1 + RungeKutta5Adapt::a52 * this->k2 + RungeKutta5Adapt::a53 * this->k3 + RungeKutta5Adapt::a54 * this->k4);
  // k5 = f(t + c5 * h, out)
  out = from + h * (RungeKutta5Adapt::a61 * this->k1 + RungeKutta5Adapt::a62 * this->k3 + RungeKutta5Adapt::a63 * this->k4 + RungeKutta5Adapt::a64 * this->k4 +
                    RungeKutta5Adapt::a65 * this->k5);
  // k6 = f(t + c6 * h, out)
  out = from + h * (RungeKutta5Adapt::b1 * this->k1 + RungeKutta5Adapt::b2 * this->k2 + RungeKutta5Adapt::b3 * this->k3 + RungeKutta5Adapt::b4 * this->k4 +
                    RungeKutta5Adapt::b5 * this->k5 + RungeKutta5Adapt::b6 * this->k6);
  return 0.;
}

void RungeKutta5Adapt::integrate(xt::xarray<double>& point, double timeStart, double timeEnd)
{
  this->integrate(point, timeStart, timeEnd, 1e-6, 1e+20, static_cast<std::int32_t>(1e+8));
}

void RungeKutta5Adapt::integrate(xt::xarray<double>& point, double timeStart, double timeEnd, double locErr)
{
  this->integrate(point, timeStart, timeEnd, locErr, 1e+20, static_cast<std::int32_t>(1e+8));
}

void RungeKutta5Adapt::integrate(
  xt::xarray<double>& point, double timeStart, double timeEnd, double locErr, double maxStep, std::int32_t maxStepCount)
{
  double error   = locErr;
  double t       = timeStart;
  double h       = (timeEnd > timeStart ? 1. : -1.) * std::min(maxStep, std::fabs(timeEnd - timeStart));
  double minStep = std::fabs(timeEnd - timeStart) / maxStepCount;
  double hn;

  xt::xarray<double> out;

  // Sanity check
  this->stepNumber = 0;

  while((t < timeEnd) && (h > 0) || (t > timeEnd) && (h < 0)) {
    error = this->step(point, t, h, out);
    hn    = (h > 0 ? 1. : -1.) *
         std::min(maxStep, std::max(minStep, std::fabs(h) * 0.95 * std::pow(locErr / error, 1. / 6.)));

    // Second 'or' operand defines an error
    // Came here from legacy C# code from where I took that integration method
    if(error < locErr || std::fabs(h) <= minStep * 1.01) {
      t += h;
      ++this->stepNumber;
      point = out;
    }

    if(h > 0) {
      h = std::min(hn, timeEnd - t);
    } else {
      h = std::max(hn, timeEnd - t);
    }
  }
}


constexpr double RungeKutta4NonAdapt::a21;
constexpr double RungeKutta4NonAdapt::a31;
constexpr double RungeKutta4NonAdapt::a32;
constexpr double RungeKutta4NonAdapt::a41;
constexpr double RungeKutta4NonAdapt::a42;
constexpr double RungeKutta4NonAdapt::a43;

constexpr double RungeKutta4NonAdapt::c2;
constexpr double RungeKutta4NonAdapt::c3;
constexpr double RungeKutta4NonAdapt::c4;

constexpr double RungeKutta4NonAdapt::b1;
constexpr double RungeKutta4NonAdapt::b2;
constexpr double RungeKutta4NonAdapt::b3;
constexpr double RungeKutta4NonAdapt::b4;


// RungeKutta4NonAdapt::RungeKutta4NonAdapt(std::function<xt::xarray<double>(double, xt::xarray<double>)> rhs) :
//     f_0{rhs}
// {

// }

RungeKutta4NonAdapt::RungeKutta4NonAdapt(std::function<xt::xarray<double>(size_t, xt::xarray<double>)> rhs) noexcept :
    f_1{rhs}
{

}

// void RungeKutta4NonAdapt::step(const xt::xarray<double>& from, double t, double h, xt::xarray<double>& out)
// {
//     // k1 = f(t, from)
//     k1 = this->f_0(t, from);
//     out = from + h * RungeKutta4NonAdapt::a21 * this->k1;
//     // k2 = f(t + c2 * h, out)
//     k2 = this->f_0(t + RungeKutta4NonAdapt::c2 * h, out);
//     out = from + h * RungeKutta4NonAdapt::a32 * this->k2;
//     // k3 = f(t + c3 * h, out)
//     k3 = this->f_0(t + RungeKutta4NonAdapt::c3 * h, out);
//     out = from + h * RungeKutta4NonAdapt::a43 * this->k3;
//     // k4 = f(t + c4 * h, out)
//     k4 = this->f_0(t + RungeKutta4NonAdapt::c4 * h, out);
//     out = from + h * (RungeKutta4NonAdapt::b1 * this->k1 + RungeKutta4NonAdapt::b2 * this->k2 + RungeKutta4NonAdapt::b3 * this->k3 + RungeKutta4NonAdapt::b4 * this->k4);
// }

void RungeKutta4NonAdapt::step(const xt::xarray<double>& from, size_t t, double h, xt::xarray<double>& out)
{   
    // k1 = f(t, from)
    k1 = this->f_1(t, from);
    out = from + h * RungeKutta4NonAdapt::a21 * this->k1;
    // k2 = f(t + c2 * h, out)
    k2 = this->f_1(t + static_cast<size_t>(RungeKutta4NonAdapt::c2 * h), out);
    out = from + h * RungeKutta4NonAdapt::a32 * this->k2;
    // k3 = f(t + c3 * h, out)
    k3 = this->f_1(t + static_cast<size_t>(RungeKutta4NonAdapt::c3 * h), out);
    out = from + h * RungeKutta4NonAdapt::a43 * this->k3;
    // k4 = f(t + c4 * h, out)
    k4 = this->f_1(t + static_cast<size_t>(RungeKutta4NonAdapt::c4 * h), out);
    out = from + h * (RungeKutta4NonAdapt::b1 * this->k1 + RungeKutta4NonAdapt::b2 * this->k2 + RungeKutta4NonAdapt::b3 * this->k3 + RungeKutta4NonAdapt::b4 * this->k4);
}

void RungeKutta4NonAdapt::integrate(xt::xarray<double>& point, double timeStart, double timeEnd, std::int32_t steps)
{
  double t = timeStart;
  double h = (timeEnd - timeStart) / steps;
  xt::xarray<double> out;

  for(auto i = 0; i < steps - 1; ++i, t += h){
      this->step(point[i], t, h, out);
      point[i + 1] = out();
  }
}
