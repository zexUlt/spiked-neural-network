#include "NumCpp.hpp"
#include <iostream>

int main(int argc, char** argv)
{
    nc::NdArray<double> a({{1, 2, 3}, {4, 5, 6}});

    std::cout << a[a.cSlice(), 3] << '\n';

    // for(const auto& x : a){
    //     std::cout << x << '\n';
    // }

    return 0;
}