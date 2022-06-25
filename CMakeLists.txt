cmake_minimum_required(VERSION 3.16)

project("spiked-neural-network" CXX)

set(CMAKE_CXX_STANDARD 14)

include(cmake/clang-cxx-dev-tools.cmake)

message(STATUS "Set build type to ${CMAKE_BUILD_TYPE}")

# =======================================
#      ENABLING COMPILE COMMANDS EXPORT FOR CLANG-TIDY
# =======================================
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# =======================================
#      ENABLING TIME MEASUREMENT DISPLAY
# =======================================
set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE "${CMAKE_COMMAND} -E time")
set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK "${CMAKE_COMMAND} -E time")
# =======================================

# =======================================
#      SETTING UP COMPILE OPTIONS
# =======================================
# add_compile_options(-Wall)
add_compile_options(
    "${CMAKE_CXX_FLAGS}" 
    "-Winvalid-pch" 
    "-Wunused"
)
# =======================================

# =======================================
#      SETTING UP PROJECT-SPECIFIC OPTIONS
# =======================================
option(ENABLE_HAND_TESTS "Disable linking of source code files except main.cpp in order to test external libs" OFF)
# =======================================

# =======================================
#      SUBMODULES CHECKOUT
# =======================================
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
   NOT EXISTS "${PROJECT_SOURCE_DIR}/external/xtensor/CMakeLists.txt" OR
   NOT EXISTS "${PROJECT_SOURCE_DIR}/external/xtensor-blas/CMakeLists.txt")
   message(FATAL_ERROR "The necessary submodules xtl and xtensor were not downloaded! GIT_SUBMODULE was turned off or failed. Please update submodules and try again.")
endif()
# =======================================


# =======================================
#    SEARCHING FOR BLAS AND LAPACK
# =======================================
add_compile_definitions(HAVE_CBLAS=1)

if(WIN32)
    find_package(OpenBLAS REQUIRED)
    set(BLAS_LIBRARIES ${CMAKE_INSTALL_PREFIX}${OpenBLAS_LIBRARIES})
else()
    find_package(BLAS REQUIRED)
    find_package(LAPACK REQUIRED)
endif()

message(STATUS "BLAS VENDOR:      ${BLA_VENDOR}")
message(STATUS "BLAS LIBRARIES    ${BLAS_LIBRARIES}")
# =======================================

set(EXTERNAL_DEPS ${PROJECT_SOURCE_DIR}/external)
set(SOURCE_CODE_DIR ${PROJECT_SOURCE_DIR}/src)
set(SOURCE_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/include)


add_subdirectory(${EXTERNAL_DEPS}/xtl)
add_subdirectory(${EXTERNAL_DEPS}/xtensor)
add_subdirectory(${EXTERNAL_DEPS}/xtensor-blas)



list(APPEND PROJECT_SOURCES main.cpp)

# =======================================
#    CHOOSING BETWEEN FULL SOURCE BUILD OR JUST main.cpp
# =======================================
if(NOT ENABLE_HAND_TESTS)
    message(STATUS "Configuring for full build")
    file(GLOB files
        # RELATIVE ${PROJECT_SOURCE_DIR}
        ${SOURCE_CODE_DIR}/*.cpp
        ${SOURCE_INCLUDE_DIR}/*.hpp
    )

    # Traversing through ./src/ and ./include/ directories and collecting source code files
    foreach(file ${files})
        list(APPEND PROJECT_SOURCES ${file})
    endforeach()
else()
    message(STATUS "Configuring for build only with main.cpp")
endif()

add_executable(
    ${PROJECT_NAME} 
    ${PROJECT_SOURCES}
) 


list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)
list(APPEND CMAKE_INCLUDE_PATH ${EXTERNAL_DEPS}/xtl/include)
list(APPEND CMAKE_INCLUDE_PATH ${EXTERNAL_DEPS}/xtensor/include)
list(APPEND CMAKE_INCLUDE_PATH ${EXTERNAL_DEPS}/xtensor-blas/include)

target_include_directories(${PROJECT_NAME}
    PUBLIC
        ${EXTERNAL_DEPS}/xtl/include
        ${EXTERNAL_DEPS}/xtensor/include
        ${EXTERNAL_DEPS}/xtensor-blas/include
    PUBLIC
        ${SOURCE_INCLUDE_DIR}
)

target_link_directories(${PROJECT_NAME}
    PUBLIC 
        ${EXTERNAL_DEPS}/xtl
        ${EXTERNAL_DEPS}/xtensor
        ${EXTERNAL_DEPS}/xtensor-blas
    PUBLIC
        ${SOURCE_CODE_DIR}
)

target_link_libraries(${PROJECT_NAME} 
    INTERFACE 
        xtensor
        xtensor::optimize
        xtensor-blas
    PUBLIC 
        ${BLAS_LIBRARIES}
        ${LAPACK_LIBRARIES}
)

# =======================================
#    SETTING UP PRECOMPILED HEADERS
# =======================================
target_precompile_headers(${PROJECT_NAME}
    PRIVATE
        ${SOURCE_INCLUDE_DIR}/precompiled.hpp
)
# =======================================