#!/bin/bash

RECONFIG=0
CMAKE_PARAMETERS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        -r|--reconfigure)
            RECONFIG=1
            shift # past option
            ;;
        -h|--help)
            echo "Configuration script."
            echo "Options:             "
            echo "       -r, --reconfigure = Generate configuration from scratch. DEFAULT=OFF."
            echo "       -h, --help        = Show this help and exit."
            exit 0
            ;;
        -D*)
            CMAKE_PARAMETERS+=( $1 )
            shift
            ;; 
        -*|--*)
            echo "Unknown option $1"
            exit 1
            ;;
    esac
done

if [[ ${RECONFIG} -eq 1 ]]; then
    echo "Cleaning previous configuration data..."
    rm -r ./build
fi

echo "CMAKE_PARAMETERS = [$CMAKE_PARAMETERS]"
echo "Building..."
cmake $CMAKE_PARAMETERS -S . -B ./build 
