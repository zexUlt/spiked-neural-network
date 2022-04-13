cmake_minimum_required(VERSION 3.23)

project("spiked-neural-network" CXX)

set(CMAKE_CXX_STANDARD 14)

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
add_definitions(-DHAVE_CBLAS=1)

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

option(ENABLE_HAND_TESTS "Disable linking of source code files except main.cpp in order to test external libs" OFF)

list(APPEND SOURCES main.cpp)

if(NOT ENABLE_HAND_TESTS)
    message(STATUS "Set ENABLE_HAND_TESTS to ${ENABLE_HAND_TESTS}")
    file(GLOB files
        # RELATIVE ${PROJECT_SOURCE_DIR}
        ${SOURCE_CODE_DIR}/*.cpp
        ${SOURCE_INCLUDE_DIR}/*.hpp
    )

    # Traversing through ./src/ and ./include/ directories and collecting source code files
    foreach(file ${files})
        list(APPEND SOURCES ${file})
    endforeach()
endif()

add_executable(
    ${PROJECT_NAME} 
    ${SOURCES}
) 


list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)
list(APPEND CMAKE_INCLUDE_PATH ${EXTERNAL_DEPS}/xtl/include)
list(APPEND CMAKE_INCLUDE_PATH ${EXTERNAL_DEPS}/xtensor/include)
list(APPEND CMAKE_INCLUDE_PATH ${EXTERNAL_DEPS}/xtensor-blas/include)

target_include_directories(${PROJECT_NAME}
    PUBLIC ${EXTERNAL_DEPS}/xtl/include
        ${EXTERNAL_DEPS}/xtensor/include
        ${EXTERNAL_DEPS}/xtensor-blas/include
        ${SOURCE_INCLUDE_DIR}
)

target_link_directories(${PROJECT_NAME}
    PUBLIC ${EXTERNAL_DEPS}/xtl
        ${EXTERNAL_DEPS}/xtensor
        ${EXTERNAL_DEPS}/xtensor-blas
        ${SOURCE_CODE_DIR}
)

target_link_libraries(${PROJECT_NAME} 
    INTERFACE xtensor
        xtensor-blas
    PUBLIC ${BLAS_LIBRARIES}
        ${LAPACK_LIBRARIES}
)
