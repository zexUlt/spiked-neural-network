#include "NumCpp.hpp"
#include "SpikeDNNet.hpp"
#include "IzhikevichActivation.hpp"
#include <iostream>

int main(int argc, char** argv)
{
    const nc::uint32 dim = 2u;

    CxxSDNN::IzhikevichActivation izh_act_1(dim);
    CxxSDNN::IzhikevichActivation izh_act_2(dim);

    CxxSDNN::SpikeDNNet dnn(
        izh_act_1, izh_act_2, // Activation functions
        20. * nc::ones<double>(2), 20. * nc::ones<double>(2), // W_1, W_2
        dim, 20. * nc::diag<double>({-1., -2.}), // dim, mat_A
        1575.9 * nc::diag<double>({60., 40,}), .15 * nc::diag<double>({10., 1.}), // mat_P, K_1
        .15 * nc::diag<double>({1., 1.}) // K_2
    );
    
    return 0;
}