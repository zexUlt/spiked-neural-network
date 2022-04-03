#pragma once

#include "NumCpp.hpp"
#include "SpikeDNNet.hpp"
#include "matplot/matplot.h"

#include <map>
#include <string>
#include <iosfwd>

namespace plt = matplot;

class UtilityFunctionLibrary
{
public:
    template<typename dtype>
    using vl_tr_map = std::map< std::string, std::pair< nc::NdArray<dtype>, nc::NdArray<dtype> > >;

    struct ValidationResults
    {
        std::pair<double, double> mse_res;
        std::pair<double, double> mae_res;
        std::pair<double, double> smae_res;

        friend std::ostream& operator<<(std::ostream& out, UtilityFunctionLibrary::ValidationResults res)
        {
            out << "MSE: [Train: " << res.mse_res.first << ", Test:" << res.mse_res.second << "]\n";
            out << "MAE: [Train: " << res.mae_res.first << ", Test:" << res.mae_res.second << "]\n";
            out << "sMAE: [Train: " << res.smae_res.first << ", Test:" << res.smae_res.second << "]\n";

            return out;
        }

    };


    template<typename dtype>
    static nc::NdArray<dtype> convolveValid(const nc::NdArray<dtype>& f, const nc::NdArray<dtype>& g)
    {
        const auto nf = f.size();
        const auto ng = g.size();
        const auto& min_v = (nf < ng) ? f : g;
        const auto& max_v = (nf < ng) ? g : f;
        const auto n = std::max(nf, ng) - std::min(nf, ng) + 1;
        nc::NdArray<dtype> out(1, n);

        for(auto i(0u); i < n; ++i){
            for(int j(min_v.size() - 1), k(i); j >=0; --j, ++k){
                out.at(i) += min_v[j] * max_v[k];
            }
        }

        return out;
    }

    template<typename dtype>
    static nc::DataCube<dtype> construct_fill_DC(const nc::NdArray<dtype>& init_val, nc::uint32 capacity)
    {
        nc::DataCube<dtype> out(capacity);

        for(int i = 0; i < capacity; ++i){
            out.push_back(init_val);
        }

        return out;
    }

    template<typename dtype>
    static ValidationResults dnn_validate(CxxSDNN::SpikeDNNet& dnn, vl_tr_map<dtype> folds)
    {
        ValidationResults results;        

        auto tr_target = folds["tr"].first;
        auto tr_control = folds["tr"].second;

        auto vl_target = folds["vl"].first;
        auto vl_control = folds["vl"].second;

        auto target_est = dnn.fit(tr_target, tr_control);

        auto vl_pred = dnn.predict(nc::NdArray<double>({target_est(target_est.shape().rows - 1, 0)}), vl_control);

        results.mse_res = std::make_pair(
            UtilityFunctionLibrary::mean_squared_error(tr_target(tr_target.rSlice(), 0), target_est(target_est.rSlice(), 0)), 
            UtilityFunctionLibrary::mean_squared_error(vl_target(vl_target.rSlice(), 0), vl_pred(vl_pred.rSlice(), 0))
        );

        results.mae_res = std::make_pair(
            UtilityFunctionLibrary::mean_absolute_error(tr_target(tr_target.rSlice(), 0), target_est(target_est.rSlice(), 0)),
            UtilityFunctionLibrary::mean_absolute_error(vl_target(vl_target.rSlice(), 0), vl_pred(vl_pred.rSlice(), 0))
        );

        results.smae_res = std::make_pair(
            results.mae_res.first / nc::mean(tr_target(tr_target.rSlice(), 0))[0],
            results.mae_res.second / nc::mean(vl_target(vl_target.rSlice(), 0))[0]
        );

        return results;
    }

    template<typename dtype>
    static double mean_squared_error(nc::NdArray<dtype> y_true, nc::NdArray<dtype> y_pred)
    {
        auto output_errors = nc::average(nc::square(y_true - y_pred), nc::Axis::ROW);

        return nc::average(output_errors)[0];
    }

    template<typename dtype>
    static double mean_absolute_error(nc::NdArray<dtype> y_true, nc::NdArray<dtype> y_pred)
    {
        auto output_errors = nc::average(nc::abs(y_pred - y_true), nc::Axis::ROW);

        return nc::average(output_errors)[0];
    }

    static void plot_article(
        nc::int32 idx, nc::NdArray<double> time, nc::int32 split,
        nc::int32 width, nc::NdArray<double> tr_target, nc::NdArray<double> tr_control,
        nc::NdArray<double> vl_target, nc::NdArray<double> vl_control, 
        nc::NdArray<double> tr_est, nc::NdArray<double> error,
        std::vector<nc::DataCube<double>> weights_dyn)
    {
        std::vector<plt::axes_handle> ax;
        for (size_t i = 0; i < 4; ++i) {
            ax.emplace_back(plt::subplot(2, 2, i));
        }

        plt::subplot(ax[0]);
        plt::plot(
            (time({0, split}, time.cSlice()) / 110.).toStlVector(), 
            nc::degrees(tr_control).toStlVector()
        )->line_width(2.).color({0., 0., 255.});

        plt::subplot(ax[1]);
        plt::scatter(
            (time({0, split}, time.cSlice()) / 110.).toStlVector(),
            (nc::degrees(tr_target(tr_target.rSlice(), 0)) + 2.).toStlVector()
        )->line_style("solid").color({165./255, 172./255, 175./255});

        plt::subplot(ax[1]);
        plt::plot(
            (time({0, split}, time.cSlice()) / 110.).toStlVector(),
            (nc::degrees(tr_est(tr_est.rSlice(), 0)) + 2.).toStlVector()
        )->line_width(2.).color({0., 0., 255.});

        plt::subplot(ax[2]);
        plt::plot(
            (time({0, split}, time.cSlice()) / 110.).toStlVector(),
            nc::degrees(error).toStlVector()
        )->line_width(2.).color({0., 0., 255.});

        plt::subplot(ax[3]);
        plt::plot(
            (time({0, split - 2}, time.cSlice()) / 110.).toStlVector(),
            nc::abs(weights_dyn[0].sliceZAll(1, {0, split-2})).toStlVector()
        )->line_width(2.).color({165./255, 172./255, 175./255});
        
        plt::subplot(ax[3]);
        plt::plot(
            (time({0, split - 2}, time.cSlice()) / 110.).toStlVector(),
            nc::abs(weights_dyn[1].sliceZAll(1, {0, split-2})).toStlVector()
        )->line_width(2.).color({0., 0., 255.});
    }

};
