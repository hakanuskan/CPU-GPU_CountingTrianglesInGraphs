#!/bin/bash

# Compile and run combination_gen_hip.cpp
hipcc -o generator combination_gen_hip.cpp
./generator

# Compile and run G1_hip.cpp through G8_hip.cpp
for i in {1..8}; do
    hipcc -o G$i G${i}_hip.cpp
    ./G$i
done

