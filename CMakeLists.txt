cmake_minimum_required(VERSION 3.22)

project("spiked-neural-network" CXX)

set(CMAKE_CXX_STANDARD 14)


add_executable(
    ${PROJECT_NAME} 
    main.cpp 
    Utility.hpp
    AbstractActivation.hpp 
    IzhikevichActivation.cpp IzhikevichActivation.hpp
    SigmoidActivation.cpp SigmoidActivation.hpp
    SpikeDNNet.cpp SpikeDNNet.hpp
)

find_package(NumCpp 2.6.2 REQUIRED)
target_link_libraries(${PROJECT_NAME} PUBLIC NumCpp::NumCpp)