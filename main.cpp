#include "NumCpp.hpp"
#include "SpikeDNNet.hpp"
#include "IzhikevichActivation.hpp"
#include "Utility.hpp"

#include "debug_header.hpp"

int main(int argc, char** argv)
{
    // nc::NdArray<int> a{{1,2}, {3,4}};
    // nc::NdArray<int> b{{1,1}, {2,2}};
    // nc::DataCube<int> c;

    // c.push_back(b);
    // c.push_back(a);

    // std::cout << nc::diff(c.sliceZAll(-1, 0), nc::Axis::ROW);

    auto tr_target  = nc::fromfile<double>("../train_data/tr_target.dmp").reshape(-1, 2);
    auto tr_control = nc::fromfile<double>("../train_data/tr_control.dmp").reshape(-1, 2);
    auto vl_target  = nc::fromfile<double>("../train_data/vl_target.dmp").reshape(-1, 2);
    auto vl_control = nc::fromfile<double>("../train_data/vl_control.dmp").reshape(-1, 2);

    // auto tr_target  = nc::fromfile<double>("train_data/tr_target.dmp").reshape(-1, 2);
    // auto tr_control = nc::fromfile<double>("train_data/tr_control.dmp").reshape(-1, 2);
    // auto vl_target  = nc::fromfile<double>("train_data/vl_target.dmp").reshape(-1, 2);
    // auto vl_control = nc::fromfile<double>("train_data/vl_control.dmp").reshape(-1, 2);

    const nc::uint32 width = 4394;
    const nc::int32 split = 3305;
    const nc::uint32 dim = 2u;

    auto izh_act_1 = new CxxSDNN::IzhikevichActivation(dim);
    auto izh_act_2 = new CxxSDNN::IzhikevichActivation(dim);

    auto W_1 = nc::ones<double>(2) * 20. ;
    auto W_2 = 20. * nc::ones<double>(2);
    auto A   = 20. * nc::diag<double>({-1., -2.});
    auto P   = 1575.9 * nc::diag<double>({60., 40,});
    auto K_1 = .15 * nc::diag<double>({10., 1.});
    auto K_2 = .15 * nc::diag<double>({1., 1.});  

    CxxSDNN::SpikeDNNet dnn(
        izh_act_1, izh_act_2, // Activation functions
        W_1, W_2, // W_1, W_2
        dim, A, // dim, mat_A
        P, K_1, // mat_P, K_1
        K_2 // K_2
    );

    UtilityFunctionLibrary::vl_tr_map<double> folds{
        {
            "tr", {tr_target, tr_control}
        }, 
        {
            "vl", {vl_target, vl_control}
        }
    };

    auto res = UtilityFunctionLibrary::dnn_validate(dnn, folds);

    std::cout << res;

    // auto time_vec = plt::linspace(0., static_cast<double>(width), width);
    // auto time = nc::NdArray<double>(time_vec.begin(), time_vec.end());
    // auto error = nc::abs(tr_target(tr_target.rSlice(), 0) - res.tr_est[0](res.tr_est[0].rSlice(), 0));
    // nc::DataCube<double> dummy;

    // // (time(time.rSlice(), {0, split}) / 110.).tofile("../plot_data/timeline.dmp", ';');
    // error.tofile("../plot_data/error.dmp", ';');
    // (nc::degrees(tr_target(tr_target.rSlice(), 0)) + 2.).tofile("../plot_data/target.dmp", ';');
    // nc::degrees(tr_control).tofile("../plot_data/control.dmp", ';');
    // (nc::degrees(res.tr_est[0](res.tr_est[0].rSlice(), 0)) + 2.).tofile("../plot_data/estimation.dmp", ';');
    // UtilityFunctionLibrary::plot_article(
    //     0, time, split, 
    //     tr_target, tr_control, res.tr_est[0],
    //     error, {dummy});

    return 0;
}