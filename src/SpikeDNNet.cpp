#include "AbstractActivation.hpp"
#include "SpikeDNNet.hpp"
#include "Utility.hpp"

#include <xtensor/xview.hpp>
#include <xtensor-blas/xlinalg.hpp>

#include "debug_header.hpp"


using CxxSDNN::SpikeDNNet;

SpikeDNNet::SpikeDNNet(
        std::unique_ptr<AbstractActivation> act_func_1,
        std::unique_ptr<AbstractActivation> act_func_2,
        xt::xarray<double> _mat_W_1,
        xt::xarray<double> _mat_W_2,
        size_t dim,
        xt::xarray<double> _mat_A,
        xt::xarray<double> _mat_P,
        xt::xarray<double> _mat_K_1,
        xt::xarray<double> _mat_K_2
    ) :
    afunc_1{std::move(act_func_1)}, afunc_2{std::move(act_func_2)}, mat_A{_mat_A},
    mat_P{_mat_P}, mat_K_1{_mat_K_1}, mat_K_2{_mat_K_2},
    init_mat_W_1{_mat_W_1}, init_mat_W_2{_mat_W_2}, mat_dim{dim}
{

}

SpikeDNNet::SpikeDNNet(const SpikeDNNet& other) noexcept
{
    *this = other;
}

SpikeDNNet& SpikeDNNet::operator=(const SpikeDNNet& other) noexcept
{
   this->mat_A = other.mat_A;
   this->mat_P = other.mat_P;
   this->mat_K_1 = other.mat_K_1;
   this->mat_K_2 = other.mat_K_2;
   this->mat_W_1 = other.mat_W_1;
   this->mat_W_2 = other.mat_W_2;
   this->init_mat_W_1 = other.init_mat_W_1;
   this->init_mat_W_2 = other.init_mat_W_2;
   this->array_hist_W_1 = other.array_hist_W_1;
   this->array_hist_W_2 = other.array_hist_W_2;
   this->smoothed_W_1 = other.smoothed_W_1;
   this->smoothed_W_2 = other.smoothed_W_2;

   return *this;
}

xt::xarray<double> SpikeDNNet::moving_average(xt::xarray<double> x, std::uint32_t w)
{
    return UtilityFunctionLibrary::convolveValid(x, xt::ones<double>({w})) / static_cast<double>(w);
}

xt::xarray<double> SpikeDNNet::smooth(xt::xarray<double> x, std::uint32_t w)
{
    auto l = x.shape(0);
    auto m = x.shape(1);
    auto n = x.shape(2);
    auto new_sizeZ = l - w + 1u;
    xt::xarray<double> new_x = xt::ones<double>({new_sizeZ, m, n});

    for(auto i = 0u; i < m; ++i){
        for(auto j = 0u; j < n; ++j){
            auto slice_x = xt::view(x, xt::all(), i, j);
            auto m_av = moving_average(slice_x, w);
            xt::view(new_x, xt::all(), i, j).assign(m_av);
        }
    }

    return new_x;
}

xt::xarray<double> SpikeDNNet::fit(
        xt::xarray<double> vec_x,
        xt::xarray<double> vec_u,
        double step,
        std::uint32_t n_epochs,
        std::uint32_t k_points)
{
    auto nt = vec_u.shape(0);
    xt::xarray<double> vec_est = 0.1 * xt::ones<double>({nt, this->mat_dim});

    this->mat_W_1 = this->init_mat_W_1;
    this->mat_W_2 = this->init_mat_W_2;

    this->array_hist_W_1 = xt::ones<double>({nt, this->mat_dim, this->mat_dim});
    this->array_hist_W_2 = xt::ones<double>({nt, this->mat_dim, this->mat_dim});

    this->neuron_1_hist = xt::ones<double>({nt, this->mat_dim, size_t(1)});
    this->neuron_2_hist = xt::ones<double>({nt, this->mat_dim, vec_u.shape(1)});

    for(int e = 0; e < n_epochs; ++e){
        vec_x = xt::eval(xt::flip(vec_x, 0));
        vec_u = xt::eval(xt::flip(vec_u, 0));  
        
        if(e > 1){
            this->mat_W_1 = xt::view(this->smoothed_W_1, -1);
            this->mat_W_2 = xt::view(this->smoothed_W_2, -1);
        }

        for(int i = 0; i < nt - 1; ++i){
            xt::xarray<double> current_vec_est = xt::view(vec_est, i);
            xt::xarray<double> current_vec_u   = xt::view(vec_u, i);
            xt::xarray<double> current_delta   = current_vec_est - xt::view(vec_x, i);

            auto neuron_out_1 = this->afunc_1->operator()(current_vec_est, 0.01);
            auto neuron_out_2 = this->afunc_2->operator()(current_vec_est, 0.01);

            auto vec_est_next = xt::view(vec_est, i + 1);
            vec_est_next = xt::eval(current_vec_est + step * (
                xt::squeeze(xt::linalg::dot(this->mat_A, current_vec_est)) +
                xt::squeeze(xt::linalg::dot(this->mat_W_1, neuron_out_1))) +
                xt::linalg::dot(
                    xt::linalg::dot(this->mat_W_2, neuron_out_2), // (4x4)x(4x2)
                    current_vec_u)
            );

            this->mat_W_1 -= step * (
                xt::linalg::dot(
                    xt::linalg::dot(
                        xt::linalg::dot(this->mat_K_1, this->mat_P), 
                        current_delta.reshape({-1, 1})
                    ),
                xt::transpose(neuron_out_1))
            );

            this->mat_W_2 -= step * (
                xt::linalg::dot(
                    xt::linalg::dot(
                        xt::linalg::dot(
                            xt::linalg::dot(this->mat_K_2, this->mat_P),
                            current_delta.reshape({-1, 1})
                        ),
                        current_vec_u.reshape({1, -1})
                    ),
                xt::transpose(neuron_out_2))
            );

            xt::view(this->array_hist_W_1, i).assign(this->mat_W_1);
            xt::view(this->array_hist_W_2, i).assign(this->mat_W_2);

            // xt::view(this->neuron_1_hist, i).assign(neuron_out_1);
            // xt::view(this->neuron_2_hist, i).assign(neuron_out_2);
        }

        this->smoothed_W_1 = this->smooth(this->array_hist_W_1, k_points);
        this->smoothed_W_2 = this->smooth(this->array_hist_W_2, k_points);
    }
    
    return vec_est;
}


xt::xarray<double> SpikeDNNet::predict(
        xt::xarray<double> init_state,
        xt::xarray<double> vec_u,
        double step)
{
    auto nt = vec_u.shape(0);
    xt::xarray<double> vec_est = init_state * xt::ones<double>({nt, this->mat_dim});

    auto W1 = xt::view(this->smoothed_W_1, -1);
    auto W2 = xt::view(this->smoothed_W_2, -1);

    for(auto i = 0u; i < nt - 1; ++i){
        auto cur_est = xt::view(vec_est, i);
        auto cur_u   = xt::view(vec_u, i);

        auto neuron_1_out = this->afunc_1->operator()(cur_est);
        auto neuron_2_out = this->afunc_2->operator()(cur_est);

        auto vec_est_next = xt::view(vec_est, i + 1);
        vec_est_next = cur_est + step * (
            xt::squeeze(xt::linalg::dot(this->mat_A, cur_est)) + 
            xt::squeeze(xt::linalg::dot(W1, neuron_1_out)) + 
            xt::linalg::dot(
                xt::linalg::dot(W2, neuron_2_out), 
                cur_u
            )
        );
    }

    return vec_est;
}


xt::xarray<double> SpikeDNNet::get_weights(std::uint8_t idx) const
{
    if(idx == 0){
        return this->array_hist_W_1;
    }

    if(idx == 1){
        return this->array_hist_W_2;
    }

    return xt::xarray<double>();
}

xt::xarray<double> SpikeDNNet::get_neurons_history(std::uint8_t idx) const
{
    if(idx == 0){
        return this->neuron_1_hist;
    }

    if(idx == 1){
        return this->neuron_2_hist;
    }

    return xt::xarray<double>();
}

const xt::xarray<double>& SpikeDNNet::get_A() const
{
    return this->mat_A;
}

const xt::xarray<double>& SpikeDNNet::get_P() const
{
    return this->mat_P;
}

const xt::xarray<double>& SpikeDNNet::get_K1() const
{
    return this->mat_K_1;
}

const xt::xarray<double>& SpikeDNNet::get_K2() const
{
    return this->mat_K_2;
}

const xt::xarray<double>& SpikeDNNet::get_W10() const
{
    return this->init_mat_W_1;
}

const xt::xarray<double>& SpikeDNNet::get_W20() const
{
    return this->init_mat_W_2;
}

const std::string SpikeDNNet::get_afunc_descr(size_t idx) const
{
    if(idx == 0){
        return this->afunc_1->whoami();
    }else if(idx == 1){
        return this->afunc_2->whoami();
    }else{
        return "";
    }
}
