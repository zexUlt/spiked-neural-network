#include "NumCpp.hpp"
#include "Utility.hpp"
#include <iostream>

int main(int argc, char** argv)
{
    nc::DataCube<double> dc(3305);

    auto a = UtilityFunctionLibrary::construct_fill_DC(nc::ones<int>(2), 3);
    std::cout << a.sizeZ() << '\n';

    for(auto i = 0; i < 3; ++i){
        std::cout << a[i] << " ";
    }
    
    // a.push_back(nc::NdArray<int>{{5,6}, {7,8}});
    // a.push_back(nc::NdArray<int>{{9,10}, {11,12}});

    // for(int i = 0; i < 3; ++i){
    //     std::cout << a[i] << '\n';
    // }

    // std::cout << a.sliceZAll(0, 0) << '\n' << a.sliceZAll(0,1) <<
    // '\n' << a.sliceZAll(1,0) << '\n' << a.sliceZAll(1,1) << '\n';
    // std::cout << UtilityFunctionLibrary::convolveValid(nc::NdArray<double>{3,3,3,4}, nc::NdArray<double>{5,3,1,1}) << '\n';
    
    // for(int i = 0; i < 3304; ++i){
    //     dc.push_back(nc::ones<double>(2));
    // }

    // std::cout << dc.sizeZ() << '\n';
    // std::cout << dc[0] << '\n';
    
    // nc::DataCube<double> new_x(3304);

    // for(int i = 0; i < 3304; ++i){
    //     new_x.push_back(nc::ones<double>(2));
    // }

    // for(int i = 0; i < 2; ++i){
    //     for(int j = 0; j < 2; ++j){
    //         new_x.s
    //     }
    // }

    return 0;
}