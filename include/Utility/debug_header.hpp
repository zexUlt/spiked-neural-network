#ifndef NDEBUG
  #include <iostream>
  #include <string>

  #define DEBUG_SHAPE(x) \
    std::cout << #x << " " << x.shape()[0] << " " << x.shape()[1] << " " << x.shape()[2] << "\n" << x.dimension() << "\n"

  #define DEBUG_PRINT(x) std::cout << #x << " " << x << '\n'

  #define DEBUG_WHERE std::cout << __FILE__ << ": " << __LINE__ << '\n'
#else
  #define DEBUG_SHAPE(x) ((void)0)
  #define DEBUG_PRINT(x) ((void)0)
  #define DEBUG_WHERE ((void)0)
#endif
