#include <iostream>
#include <string>

#define DEBUG_SHAPE(x) std::cout << #x << " " << x.shape()[0] << " " << x.shape()[1] << "\n" << x.dimension() << "\n"

#define DEBUG_XARRAY(x) std::cout << #x << " " << x << '\n'