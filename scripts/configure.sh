#!/bin/bash


RECONFIG=0
DISABLE_PCH=1
USE_PARALLEL=0
NJOBS=1
CMAKE_PARAMETERS=()

echo "Collecting parameters..."

CURRENT_OPTION=$1
while [[ $# -gt 0 ]]; do
    case ${CURRENT_OPTION} in
        -r|--reconfigure)
            RECONFIG=1
            shift # past option
            ;;
        -h|--help)
            echo "Configuration script."
            echo "Options:             "
            echo "       -r,   --reconfigure        = Generate configuration from scratch. DEFAULT=OFF."
            echo "       -h,   --help               = Show this help and exit."
	        echo "       -bt,  --build-type         = Select build type for the project."
            echo "       -pch, --precompile-headers = If present enables PCH for the project."
            echo "       -j,   --parallel <njobs>   = Sets <njobs> running for parallel build."
            echo ""
            echo "This script also accepts every CMake flag existing."
            exit 0
            ;;
	    -bt|--build-type)
	        CMAKE_PARAMETERS+=( "-DCMAKE_BUILD_TYPE=$2" )
            shift
            shift
            ;;
        -pch|--precompile-headers)
            CMAKE_PARAMETERS+=( "-DCMAKE_DISABLE_PRECOMPILE_HEADERS=OFF" )
            DISABLE_PCH=0
            shift
            ;;
        -D*)
            CMAKE_PARAMETERS+=( ${CURRENT_OPTION} )
            shift
            ;;
        -j|--parallel)
            USE_PARALLEL=1
            shift
            NJOBS=$1
            shift
            ;;
        -*|--*)
            echo "Unknown option ${CURRENT_OPTION}"
            CURRENT_OPTION="-h"
            ;;
    esac
    CURRENT_OPTION=$1
done

if [[ ${RECONFIG} -eq 1 ]]; then
    echo "Cleaning previous configuration data..."
    rm -r ./build
fi

if [[ ${DISABLE_PCH} -eq 1 ]]; then
    CMAKE_PARAMETERS+=( "-DCMAKE_DISABLE_PRECOMPILE_HEADERS=ON" )
fi

if [[ ${USE_PARALLEL} -eq 1 ]]; then
    if [[ ${NJOBS} -gt 0 ]]; then
        CMAKE_PARAMETERS+=( "-j ${NJOBS}" )
    else
        echo "Illegal number of jobs for parallel build. Setting back to default..."
    fi
fi

echo "CMAKE_PARAMETERS = [${CMAKE_PARAMETERS[@]}]"
echo "Building..."
cmake $CMAKE_PARAMETERS -S . -B ./build
