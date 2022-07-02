#include "SpikeDNNet.hpp"

#include "ActivationFunctions/AbstractActivation.hpp"
#include "Utility/Utility.hpp"
#include "Utility/debug_header.hpp"

using cxx_sdnn::SpikeDNNet;

SpikeDNNet::SpikeDNNet(
  std::unique_ptr<AbstractActivation> actFunc1, std::unique_ptr<AbstractActivation> actFunc2, xt::xarray<double> matW1,
  xt::xarray<double> matW2, size_t dim, xt::xarray<double> matA, xt::xarray<double> matP, xt::xarray<double> matK1,
  xt::xarray<double> matK2, double _alpha) :
  afunc1{std::move(actFunc1)},
  afunc2{std::move(actFunc2)}, matA{matA}, matP{matP}, matK1{matK1}, matK2{matK2}, initMatW1{matW1}, initMatW2{matW2},
  matDim{dim}, alpha{_alpha}
{}

SpikeDNNet::SpikeDNNet(const SpikeDNNet& other) noexcept
{
  *this = other;
}

SpikeDNNet& SpikeDNNet::operator=(const SpikeDNNet& other) noexcept
{
  this->matA        = other.matA;
  this->matP        = other.matP;
  this->matK1       = other.matK1;
  this->matK2       = other.matK2;
  this->matW1       = other.matW1;
  this->matW2       = other.matW2;
  this->initMatW1   = other.initMatW1;
  this->initMatW2   = other.initMatW2;
  this->arrayHistW1 = other.arrayHistW1;
  this->arrayHistW2 = other.arrayHistW2;
  this->smoothedW1  = other.smoothedW1;
  this->smoothedW2  = other.smoothedW2;

  return *this;
}

xt::xarray<double> SpikeDNNet::moving_average(xt::xarray<double> x, std::uint32_t w)
{
  return UtilityFunctionLibrary::convolve_valid(x, xt::ones<double>({w})) / static_cast<double>(w);
}

xt::xarray<double> SpikeDNNet::smooth(xt::xarray<double> x, std::uint32_t w)
{
  auto l        = x.shape(0);
  auto m        = x.shape(1);
  auto n        = x.shape(2);
  auto newSizeZ = l - w + 1u;

  xt::xarray<double> newX = xt::ones<double>({newSizeZ, m, n});

  for(auto i = 0u; i < m; ++i) {
    for(auto j = 0u; j < n; ++j) {
      auto sliceX = xt::view(x, xt::all(), i, j);
      auto mAv    = moving_average(sliceX, w);
      xt::view(newX, xt::all(), i, j).assign(mAv);
    }
  }

  return newX;
}

xt::xarray<double> SpikeDNNet::fit(
  xt::xarray<double> vecX, xt::xarray<double> vecU, double step, std::uint32_t nEpochs, std::uint32_t kPoints)
{
  auto nt                   = vecU.shape(0);
  xt::xarray<double> vecEst = .01 * xt::ones<double>({nt, this->matDim});

  xt::xarray<double> matWTr1 = xt::ones<double>(this->matW1.shape());
  xt::xarray<double> matWTr2 = xt::ones<double>(this->matW2.shape());

  this->matW1 = this->initMatW1;
  this->matW2 = this->initMatW2;

  this->arrayHistW1 = xt::ones<double>({nt + 1, this->matW1.shape(0), this->matW1.shape(1)});
  this->arrayHistW2 = xt::ones<double>({nt + 1, this->matW1.shape(0), this->matW1.shape(1)});

  this->deltaHist = xt::zeros<double>({nt, this->matDim});

  xt::view(this->arrayHistW1, 0).assign(this->matW1);
  xt::view(this->arrayHistW2, 0).assign(this->matW2);

  this->neuron1Hist = xt::ones<double>({nt, this->matDim, size_t(1)});
  this->neuron2Hist = xt::ones<double>({nt, this->matDim, vecU.shape(1)});

  for(std::uint32_t e = 0; e < nEpochs; ++e) {
    vecX = xt::eval(xt::flip(vecX, 0));
    vecU = xt::eval(xt::flip(vecU, 0));

    if(e > 1) {
      this->matW1 = xt::view(this->smoothedW1, -1);
      this->matW2 = xt::view(this->smoothedW2, -1);

      matWTr1 = xt::view(this->smoothedW1, -1);
      matWTr2 = xt::view(this->smoothedW2, -1);

      xt::view(this->arrayHistW1, 0).assign(this->matW1);
      xt::view(this->arrayHistW2, 0).assign(this->matW2);
    }

    for(size_t i = 0; i < nt - 1; ++i) {
      xt::xarray<double> currentVecEst = xt::view(vecEst, i);
      xt::xarray<double> currentVecU   = xt::view(vecU, i);
      xt::xarray<double> currentDelta  = currentVecEst - xt::view(vecX, i);

      auto neuronOut1 = (*this->afunc1)(currentVecEst, 0.01);
      auto neuronOut2 = (*this->afunc2)(currentVecEst, 0.01);

      auto vecEstNext = xt::view(vecEst, i + 1); // vec_est[i + 1]

      vecEstNext = xt::eval(
        currentVecEst + step * (xt::squeeze(xt::linalg::dot(this->matA, currentVecEst)) +
                                xt::squeeze(xt::linalg::dot(this->matW1, neuronOut1)) +
                                xt::linalg::dot(xt::linalg::dot(this->matW2, neuronOut2), currentVecU)));

      xt::xarray<double> matWTilde1 = matWTr1 - this->matW1;
      xt::xarray<double> matWTilde2 = matWTr2 - this->matW2;

      // Calculating right-hand sides of dWi/dt
      xt::xarray<double> rhsW1 =
        -xt::linalg::dot(
          xt::linalg::dot(xt::linalg::dot(this->matK1, this->matP), currentDelta.reshape({-1, 1})), // (4, ) -> (4, 1)
          xt::transpose(neuronOut1)) +
        this->alpha * matWTilde1;

      xt::xarray<double> rhsW2 =
        -xt::linalg::dot(
          xt::linalg::dot(
            xt::linalg::dot(xt::linalg::dot(this->matK2, this->matP), currentDelta.reshape({-1, 1})),
            currentVecU.reshape({1, -1})),
          xt::transpose(neuronOut2)) +
        this->alpha * matWTilde2;

      // Calling activation functions on the next state vector, but without
      // changing their own state. This is needed for implicit Runge-Kutta
      // integration method
      auto constNeuronOut1         = const_cast<decltype(*this->afunc1)>(*this->afunc1)(vecEstNext, 0.01);
      auto constNeuronOut2         = const_cast<decltype(*this->afunc2)>(*this->afunc2)(vecEstNext, 0.01);
      xt::xarray<double> nextDelta = vecEstNext - xt::view(vecX, i + 1);
      xt::xarray<double> nextVecU  = xt::view(vecU, i + 1);

      // Prediction
      xt::xarray<double> predictedW1 = this->matW1 - step * rhsW1;
      xt::xarray<double> predictedW2 = this->matW2 - step * rhsW2;

      // Calculating the right-hand side of dWi/dt
      // On the next sample
      matWTilde1 = matWTr1 - predictedW1;
      matWTilde2 = matWTr2 - predictedW2;

      xt::xarray<double> rhsW1NextStep =
        -xt::linalg::dot(
          xt::linalg::dot(xt::linalg::dot(this->matK1, this->matP), nextDelta.reshape({-1, 1})),
          xt::transpose(constNeuronOut1)) +
        this->alpha * matWTilde1;

      xt::xarray<double> rhsW2NextStep =
        -xt::linalg::dot(
          xt::linalg::dot(
            xt::linalg::dot(xt::linalg::dot(this->matK2, this->matP), nextDelta.reshape({-1, 1})),
            nextVecU.reshape({1, -1})),
          xt::transpose(constNeuronOut2)) +
        this->alpha * matWTilde2;

      // Correction
      this->matW1 = this->matW1 + step * (rhsW1 + rhsW1NextStep) / 2.;
      this->matW2 = this->matW2 + step * (rhsW2 + rhsW2NextStep) / 2.;

      xt::view(this->arrayHistW1, i + 1).assign(this->matW1);
      xt::view(this->arrayHistW2, i + 1).assign(this->matW2);

      xt::view(this->deltaHist, i).assign(currentDelta);

      xt::view(this->neuron1Hist, i).assign(neuronOut1);
      xt::view(this->neuron2Hist, i).assign(neuronOut2);
    }

    this->smoothedW1 = this->smooth(this->arrayHistW1, kPoints);
    this->smoothedW2 = this->smooth(this->arrayHistW2, kPoints);
  }

  return vecEst;
}

xt::xarray<double> SpikeDNNet::predict(xt::xarray<double> initState, xt::xarray<double> vecU, double step)
{
  auto nt                   = vecU.shape(0);
  xt::xarray<double> vecEst = initState * xt::ones<double>({nt, this->matDim});

  auto w1 = xt::view(this->smoothedW1, -1);
  auto w2 = xt::view(this->smoothedW2, -1);

  for(auto i = 0u; i < nt - 1; ++i) {
    auto curEst = xt::view(vecEst, i);
    auto curU   = xt::view(vecU, i);

    auto neuron1Out = this->afunc1->operator()(curEst);
    auto neuron2Out = this->afunc2->operator()(curEst);

    auto vecEstNext = xt::view(vecEst, i + 1);
    vecEstNext =
      curEst + step * (xt::squeeze(xt::linalg::dot(this->matA, curEst)) + xt::squeeze(xt::linalg::dot(w1, neuron1Out)) +
                       xt::linalg::dot(xt::linalg::dot(w2, neuron2Out), curU));
  }

  return vecEst;
}

double SpikeDNNet::integral_loss()
{
  double loss{0.};

  auto w1Mean = xt::mean(this->arrayHistW1, 0);
  auto w2Mean = xt::mean(this->arrayHistW2, 0);
  auto timeSamples = this->arrayHistW1.shape(0);

  for(auto i = 0u; i < timeSamples; ++i){
    auto dW1 = xt::view(this->arrayHistW1, i) - w1Mean;
    auto trace1 = xt::sum(
      xt::diagonal(
        xt::linalg::dot(
          xt::transpose(dW1),
          dW1)))();
    
    auto dW2 = xt::view(this->arrayHistW2, i) - w2Mean;
    auto trace2 = xt::sum(
      xt::diagonal(
        xt::linalg::dot(
          xt::transpose(dW2),
          dW2)))();
    
    auto pDelta = xt::linalg::dot(
      xt::linalg::dot(
        xt::transpose(xt::view(this->deltaHist, i)),
        this->matP),
      xt::view(this->deltaHist, i))();

    loss += pDelta + trace1 + trace2;
  }

  loss /= timeSamples;

  return loss;
}


xt::xarray<double> SpikeDNNet::get_weights(std::uint8_t idx) const
{
  if(idx == 0) {
    return this->arrayHistW1;
  }

  if(idx == 1) {
    return this->arrayHistW2;
  }

  return xt::xarray<double>();
}

xt::xarray<double> SpikeDNNet::get_neurons_history(std::uint8_t idx) const
{
  if(idx == 0) {
    return this->neuron1Hist;
  }

  if(idx == 1) {
    return this->neuron2Hist;
  }

  return xt::xarray<double>();
}

const xt::xarray<double>& SpikeDNNet::get_A() const
{
  return this->matA;
}

const xt::xarray<double>& SpikeDNNet::get_P() const
{
  return this->matP;
}

const xt::xarray<double>& SpikeDNNet::get_K1() const
{
  return this->matK1;
}

const xt::xarray<double>& SpikeDNNet::get_K2() const
{
  return this->matK2;
}

const xt::xarray<double>& SpikeDNNet::get_W10() const
{
  return this->initMatW1;
}

const xt::xarray<double>& SpikeDNNet::get_W20() const
{
  return this->initMatW2;
}

const std::string SpikeDNNet::get_afunc_descr(size_t idx) const
{
  if(idx == 0) {
    return this->afunc1->whoami();
  } else if(idx == 1) {
    return this->afunc2->whoami();
  } else {
    return "";
  }
}
