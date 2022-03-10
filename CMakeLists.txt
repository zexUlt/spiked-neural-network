set(CMAKE_MINIMUM_REQUIRED_VERSION 3.22)

project("spiked-neural-network" CXX)

set(CMAKE_CXX_STANDARD 14)

add_executable(${PROJECT_NAME} main.cpp)

find_package(NumCpp 2.6.2 REQUIRED)
target_link_libraries(${PROJECT_NAME} NumCpp::NumCpp)