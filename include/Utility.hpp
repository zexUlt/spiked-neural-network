#pragma once

#include "SpikeDNNet.hpp"

#include <xtensor/xview.hpp>
#include "debug_header.hpp"

#include <map>
#include <string>
#include <iosfwd>


class UtilityFunctionLibrary
{
public:
    template<typename dtype>
    using vl_tr_map = std::map< std::string, std::pair< xt::xarray<dtype>, xt::xarray<dtype> > >;

    struct ValidationResults
    {
        std::pair<double, double> mse_res;
        std::pair<double, double> mae_res;
        std::pair<double, double> smae_res;
        std::vector<xt::xarray<double>> tr_est;
        std::vector<xt::xarray<double>> vl_res;
        xt::xarray<double> W_1;
        xt::xarray<double> W_2;

        friend std::ostream& operator<<(std::ostream& out, UtilityFunctionLibrary::ValidationResults res)
        {
            out << "MSE: [Train: " << res.mse_res.first << ", Test:" << res.mse_res.second << "]\n";
            out << "MAE: [Train: " << res.mae_res.first << ", Test:" << res.mae_res.second << "]\n";
            out << "sMAE: [Train: " << res.smae_res.first << ", Test:" << res.smae_res.second << "]\n";

            return out;
        }

    };


    template<typename dtype>
    static xt::xarray<dtype> convolveValid(const xt::xarray<dtype>& f, const xt::xarray<dtype>& g)
    {
        const auto nf = f.size();
        const auto ng = g.size();
        
        const auto& min_v = (nf < ng) ? f : g;
        const auto& max_v = (nf < ng) ? g : f;
        const auto n = std::max(nf, ng) - std::min(nf, ng) + 1;
        std::vector<size_t> shape{n};
        xt::xarray<dtype> out(shape);

        for(auto i(0u); i < n; ++i){
            for(int j(min_v.size() - 1), k(i); j >= 0; --j, ++k){
                out.at(i) += min_v[j] * max_v[k];
            }
        }

        return out;
    }


    static ValidationResults dnn_validate(CxxSDNN::SpikeDNNet& dnn, vl_tr_map<double> folds, std::uint16_t n_epochs, std::uint16_t k_points)
    {
        ValidationResults results;        

        auto tr_target = folds["tr"].first;
        auto tr_control = folds["tr"].second;

        auto vl_target = folds["vl"].first;
        auto vl_control = folds["vl"].second;

        auto target_est = dnn.fit(tr_target, tr_control, 0.01, n_epochs, k_points);
        auto vl_pred = dnn.predict(xt::view(tr_target, -1, 0), vl_control);

        xt::xarray<double> tr_col     = xt::col(tr_target, 0);
        xt::xarray<double> target_col = xt::col(target_est, 0);
        xt::xarray<double> vl_col     = xt::col(vl_target, 0);
        xt::xarray<double> pred_col   = xt::col(vl_pred, 0);
        results.mse_res = std::make_pair(
            UtilityFunctionLibrary::mean_squared_error<double>(tr_col, target_col), 
            UtilityFunctionLibrary::mean_squared_error<double>(vl_col, pred_col)
        );

        results.mae_res = std::make_pair(
            UtilityFunctionLibrary::mean_absolute_error<double>(tr_col, target_col),
            UtilityFunctionLibrary::mean_absolute_error<double>(vl_col, pred_col)
        );

        results.smae_res = std::make_pair(
            (results.mae_res.first / xt::mean(tr_col)()),
            (results.mae_res.second / xt::mean(vl_col)())
        );

        results.tr_est.emplace_back(target_est);
        results.vl_res.emplace_back(vl_pred);
        results.W_1 = dnn.get_weights(0);
        results.W_2 = dnn.get_weights(1);

        return results;
    }

    template<typename dtype>
    static double mean_squared_error(xt::xarray<dtype> y_true, xt::xarray<dtype> y_pred)
    {
        xt::xarray<dtype> sq = xt::square(y_true - y_pred);
        auto output_errors = xt::average(sq);

        return xt::average(output_errors)();
    }

    template<typename dtype>
    static double mean_absolute_error(xt::xarray<dtype> y_true, xt::xarray<dtype> y_pred)
    {
        xt::xarray<dtype> absol = xt::abs(y_pred - y_true);
        auto output_errors = xt::average(absol);

        return xt::average(output_errors)();
    }
};
