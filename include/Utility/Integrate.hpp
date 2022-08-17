#pragma once

#include "precompiled.hpp"

namespace cxx_sdnn
{
    class RungeKutta5Adapt
    {
    public:
        RungeKutta5Adapt();

        double step(const xt::xarray<double>& from, double t, double h, xt::xarray<double>& out);
        void integrate(xt::xarray<double>& point, double timeStart, double timeEnd);
        void integrate(xt::xarray<double>& point, double timeStart, double timeEnd, double locErr);
        void integrate(xt::xarray<double>& point, double timeStart, double timeEnd, double locErr, double maxStep, std::int32_t maxStepCount);

        std::int32_t stepNumber;

    private:
        static constexpr double a21 = 1.0 / 5.0;
        static constexpr double a31 = 3.0 / 40.0;
        static constexpr double a32 = 9.0 / 40.0;
        static constexpr double a41 = 44.0 / 45.0;
        static constexpr double a42 = -56.0 / 15.0;
        static constexpr double a43 = 32.0 / 9.0;
        static constexpr double a51 = 19372.0 / 6561.0;
        static constexpr double a52 = -25360.0 / 2187.0;
        static constexpr double a53 = 64448.0 / 6561.0;
        static constexpr double a54 = -212.0 / 729.0;
        static constexpr double a61 = 9017.0 / 3168.0;
        static constexpr double a62 = -355.0 / 33.0;
        static constexpr double a64 = 49.0 / 176.0;
        static constexpr double a63 = 46732.0 / 5247.0;
        static constexpr double a65 = -5103.0 / 18656.0;
        static constexpr double a71 = 35.0 / 384.0;
        static constexpr double a72 = 0.0;
        static constexpr double a73 = 500.0 / 1113.0;
        static constexpr double a74 = 125.0 / 192.0;
        static constexpr double a75 = -2187.0 / 6784.0;
        static constexpr double a76 = 11.0 / 84.0;

        static constexpr double c2 = 1.0 / 5.0;
        static constexpr double c3 = 3.0 / 10.0;
        static constexpr double c4 = 4.0 / 5.0;
        static constexpr double c5 = 8.0 / 9.0;
        static constexpr double c6 = 1.0;
        static constexpr double c7 = 1.0;

        static constexpr double b1 = 5179.0 / 57600.0;
        static constexpr double b2 = 0.0;
        static constexpr double b3 = 7571.0 / 16695.0;
        static constexpr double b4 = 393.0 / 640.0;
        static constexpr double b5 = -92097.0 / 339200.0;
        static constexpr double b6 = 187.0 / 2100.0;
        static constexpr double b7 = 1.0 / 40.0;

        xt::xarray<double> k1;
		xt::xarray<double> k2;
		xt::xarray<double> k3;
		xt::xarray<double> k4;
        xt::xarray<double> k5;
        xt::xarray<double> k6;
        xt::xarray<double> k7;


    };


    class RungeKutta4NonAdapt
    {
    public:
        // explicit RungeKutta4NonAdapt(std::function<xt::xarray<double>(double, xt::xarray<double>)> rhs) noexcept;
        explicit RungeKutta4NonAdapt(std::function<xt::xarray<double>(size_t, xt::xarray<double>)> rhs) noexcept;

        // void step(const xt::xarray<double>& from, double t, double h, xt::xarray<double>& out);
        void step(const xt::xarray<double>& from, size_t t, double h, xt::xarray<double>& out);
        void integrate(xt::xarray<double>& point, double timeStart, double timeEnd, std::int32_t steps);
    
        static constexpr double a21 = 0.5;
        static constexpr double a31 = 0.0;
        static constexpr double a32 = 0.5;
        static constexpr double a41 = 0.0;
        static constexpr double a42 = 0.0;
        static constexpr double a43 = 1.0;

        static constexpr double c2 = 0.5;
        static constexpr double c3 = 0.5;
        static constexpr double c4 = 1.0;

        static constexpr double b1 = 1.0 / 6.0;
        static constexpr double b2 = 1.0 / 3.0;
        static constexpr double b3 = 1.0 / 3.0;
        static constexpr double b4 = 1.0 / 6.0;

    private:
        // std::function<xt::xarray<double>(double, xt::xarray<double>)> f_0;
        std::function<xt::xarray<double>(size_t, xt::xarray<double>)> f_1;

        xt::xarray<double> k1;
        xt::xarray<double> k2;
        xt::xarray<double> k3;
        xt::xarray<double> k4;
    };
}; // namespace cxx_sdnn