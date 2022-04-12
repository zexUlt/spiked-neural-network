#!/bin/sh

RECONFIG=0

if [ $# -gt 0 ]; then
    case $1 in
        -r|--reconfigure)
            RECONFIG=1
            ;;
        -h|--help)
            echo "Configuration script."
            echo "Options:             "
            echo "       -r, --reconfigure = Generate configuration from scratch. DEFAULT=OFF."
            echo "       -h, --help        = Show this help and exit."
            exit 0
            ;;
        -*|--*)
            echo "Unknown option $1"
            exit 1
            ;;
    esac
fi

if [ ${RECONFIG} -eq 1 ]; then
    echo "Cleaning previous configuration data..."
    rm -r ./build
fi

echo "Building..."
cmake -S . -B ./build
