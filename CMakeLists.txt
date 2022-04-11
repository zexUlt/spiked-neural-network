cmake_minimum_required(VERSION 3.22)

project("spiked-neural-network" CXX)

set(CMAKE_CXX_STANDARD 14)


add_executable(
    ${PROJECT_NAME} 
    main.cpp) 
    # Utility.hpp
    # AbstractActivation.hpp 
    # IzhikevichActivation.cpp IzhikevichActivation.hpp
    # SigmoidActivation.cpp SigmoidActivation.hpp
    # SpikeDNNet.cpp SpikeDNNet.hpp
# )

find_package(Git QUIET)
if(GIT_FOUND AND EXISTS "${PROJECT_SOURCE_DIR}/.git")
# Update submodules as needed
    option(GIT_SUBMODULE "Check submodules during build" ON)
    if(GIT_SUBMODULE)
        message(STATUS "Submodule update")
        execute_process(COMMAND ${GIT_EXECUTABLE} submodule update --init --recursive
                        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                        RESULT_VARIABLE GIT_SUBMOD_RESULT)
        if(NOT GIT_SUBMOD_RESULT EQUAL "0")
            message(FATAL_ERROR "git submodule update --init failed with ${GIT_SUBMOD_RESULT}, please checkout submodules.")
        endif()
    endif()
endif()

if(NOT EXISTS "${PROJECT_SOURCE_DIR}/external/xtl/CMakeLists.txt" OR 
   NOT EXISTS "${PROJECT_SOURCE_DIR}/external/xtensor/CMakeLists.txt")
   message(FATAL_ERROR "The necessary submodules xtl and xtensor were not downloaded! GIT_SUBMODULE was turned off or failed. Please update submodules and try again.")
endif()

set(EXTERNAL_DEPS ${CMAKE_SOURCE_DIR}/external)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)

add_subdirectory(${EXTERNAL_DEPS}/xtl)
add_subdirectory(${EXTERNAL_DEPS}/xtensor)

target_include_directories(${PROJECT_NAME}
    PUBLIC ${EXTERNAL_DEPS}/xtl/include
        ${EXTERNAL_DEPS}/xtensor/include
)

target_link_directories(${PROJECT_NAME}
    PUBLIC ${EXTERNAL_DEPS}/xtl
        ${EXTERNAL_DEPS}/xtensor
)

target_link_libraries(${PROJECT_NAME} PUBLIC xtensor)
