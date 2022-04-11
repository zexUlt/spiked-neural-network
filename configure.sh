#!/bin/sh

REBUILD=0

if [ $# -gt 0 ]; then
    case $1 in
        -r|--rebuild)
            REBUILD=1
            ;;
        -h|--help)
            echo "Configuration script."
            echo "Options:             "
            echo "       -r, --rebuild = Clean build folder before new build. DEFAULT=OFF."
            echo "       -h, --help    = Show this help and exit."
            exit 0
            ;;
        -*|--*)
            echo "Unknown option $1"
            exit 1
            ;;
    esac
fi

if [ ${REBUILD} -eq 1 ]; then
    echo "Cleaning previous build data..."
    rm -r ./build
fi

echo "Building..."
cmake -S . -B ./build
