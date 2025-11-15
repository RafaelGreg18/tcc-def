#!/bin/bash

# Check if an argument was provided
# if [ -z "$1" ]; then
#     echo "Usage: $0 <CUDA_DEVICE_IDS>"
#     exit 1
# fi

# export CUDA_VISIBLE_DEVICES=$1


for eaflweight in 0.25 0.5 0.75; do

  for numedgeservers in 25 50 100; do
    flwr run . --run-config="num-edge-servers=$numedgeservers seed=42 num-rounds=150 selection-name='eaflplus' dir-alpha=0.1 use-battery=true eafl-weight=$eaflweight"
    flwr run . --run-config="num-edge-servers=$numedgeservers seed=42 num-rounds=150 selection-name='eaflplus' dir-alpha=1.0 use-battery=true eafl-weight=$eaflweight"
  done

done