#!/bin/bash

FILES=()

while [[ $# -gt 0 ]]; do
    if [[ -f $1 ]]; then 
        FILES+=( $1 )
        shift;
    fi
done

clang-format -i ${FILES[@]}
