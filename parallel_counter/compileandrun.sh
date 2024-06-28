#!/bin/bash

# Compile and run combination_gen.cu
nvcc -o generator combination_gen.cu
./generator

# Compile and run G1.cu through G8.cu
for i in {1..8}; do
    nvcc -o G$i G$i.cu
    ./G$i
    
done
