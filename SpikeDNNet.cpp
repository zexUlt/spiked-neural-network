#include "AbstractActivation.hpp"

#include "SpikeDNNet.hpp"
#include "Utility.hpp"

using CxxSDNN::SpikeDNNet;

SpikeDNNet::SpikeDNNet(
        AbstractActivation& act_func_1,
        AbstractActivation& act_func_2,
        nc::NdArray<double> _mat_W_1,
        nc::NdArray<double> _mat_W_2,
        int dim,
        nc::NdArray<double> _mat_A,
        nc::NdArray<double> _mat_P,
        nc::NdArray<double> _mat_K_1,
        nc::NdArray<double> _mat_K_2
    ) :
    afunc_1(act_func_1), afunc_2(act_func_2), mat_A(_mat_A),
    mat_P(_mat_P), mat_K_1(_mat_K_1), mat_K_2(_mat_K_2),
    init_mat_W_1(_mat_W_1)
{

}

nc::NdArray<double> SpikeDNNet::moving_average(nc::NdArray<double> x, nc::uint32 w)
{
    return UtilityFunctionLibrary::convolveValid(x, nc::ones<double>(1, w)) / static_cast<double>(w);
}

nc::DataCube<double> SpikeDNNet::smooth(nc::DataCube<double> x, nc::uint32 w)
{
    auto l = x.sizeZ();
    auto m = x.shape().rows;
    auto n = x.shape().cols;

    auto new_x = UtilityFunctionLibrary::construct_fill_DC(nc::ones<double>(m, n), l - w + nc::uint32(1));

    for(int i = 0; i < m; ++i){
        for(int j = 0; j < n; ++j){
            auto average = this->moving_average(x.sliceZAll(i, j), w);
            for(int k = 0; k < l; ++k){
                new_x[k](i, j) = average[k];
            }
        }
    }

    return new_x;
}

nc::NdArray<double> SpikeDNNet::fit(
        nc::NdArray<double> vec_x,
        nc::NdArray<double> vec_u,
        double step,
        int n_epochs,
        int k_points)
{
    auto nt = vec_u.size();
    auto vec_est = 0.1 * nc::ones<double>(nt, this->mat_dim);

    this->mat_W_1 = this->init_mat_W_1.copy();
    this->mat_W_2 = this->init_mat_W_2.copy();

    for(int e = 1; e < n_epochs + 1; ++e){
        vec_x = nc::flip(vec_x, nc::Axis::ROW);
        vec_u = nc::flip(vec_u, nc::Axis::ROW);

        if(e > 1){
            this->mat_W_1 = this->smoothed_W_1.back().copy();
            this->mat_W_2 = this->smoothed_W_2.back().copy();
        }

        for(int i = 0; i < nt - 1; ++i){
            auto delta = vec_est - vec_x;

            auto neuron_out_1 = this->afunc_1(vec_est(i, vec_est.cSlice()), 0.01);
            auto neuron_out_2 = this->afunc_2(vec_est(i, vec_est.cSlice()), 0.01);

            vec_est(i + 1, vec_est.cSlice()) = vec_est(i, vec_est.cSlice()) + 
                step * (
                    nc::matmul(this->mat_A, vec_est(i, vec_est.cSlice())) + 
                    nc::matmul(this->mat_W_1, neuron_out_1) +
                    nc::matmul(this->mat_W_2, nc::matmul(
                        nc::diag(neuron_out_2), 
                        vec_u(i, vec_u.cSlice())
                    ))
                );
            
            this->mat_W_1 -= step * (
                nc::matmul(this->mat_K_1, 
                    nc::matmul(
                        this->mat_P, 
                        nc::matmul(
                            delta(i, delta.cSlice()), 
                            neuron_out_1
                        )
                    )
                )
            );

            this->mat_W_2 -= step * (
                nc::matmul(this->mat_K_2, 
                    nc::matmul(
                        this->mat_P, 
                        nc::matmul(
                            delta(i, delta.cSlice()), 
                            nc::matmul(
                                nc::diag(neuron_out_2),
                                vec_u(i, vec_u.cSlice())
                            )
                        )                    
                    )
                )
            );

            this->array_hist_W_1.push_back(this->mat_W_1.copy());
            this->array_hist_W_2.push_back(this->mat_W_2.copy());
        }

        this->smoothed_W_1 = this->smooth(this->array_hist_W_1, k_points);
        this->smoothed_W_2 = this->smooth(this->array_hist_W_2, k_points);
    }

    return vec_est;
}


nc::NdArray<double> SpikeDNNet::predict(
        nc::NdArray<double> init_state,
        nc::NdArray<double> vec_u,
        double step)
{
    auto nt = vec_u.size();
    auto vec_est = init_state * nc::ones<double>(nt, this->mat_dim);

    const auto& W_1 = this->smoothed_W_1.back();
    const auto& W_2 = this->smoothed_W_2.back();

    for(int i = 0; i < nt - 1; ++i){
        vec_est(i + 1, vec_est.cSlice()) = vec_est(i, vec_est.cSlice()) +
            step * (
                nc::matmul(
                    this->mat_A,
                    vec_est(i, vec_est.cSlice())
                ) + 
                nc::matmul(
                    W_1,
                    this->afunc_1(vec_est(i, vec_est.cSlice()))
                ) + 
                nc::matmul(
                    W_2,
                    nc::matmul(
                        this->afunc_2(vec_est(i, vec_est.cSlice())),
                        vec_u(i, vec_u.cSlice())
                    )
                )
            );
    }

    return vec_est;
}