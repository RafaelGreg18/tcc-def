#!/bin/bash

# Check if an argument was provided
if [ -z "$1" ]; then
    echo "Usage: $0 <CUDA_DEVICE_IDS>"
    exit 1
fi

export CUDA_VISIBLE_DEVICES=$1

# Run each configuration 3 times
for i in {1..3}; do
  # criar modelo
  python gen_sim_model.py --seed $i

  # criar perfis
  python gen_sim_cid_profile.py --seed $i

  #participantsxperformancexcost
  for participants in 5 10 20 40 80 100; do
    for alpha in 0.1 0.3 1.0; do
      flwr run . gpu-sim-dl --run-config="seed=$i num-participants=$participants dir-alpha=$alpha"
    done
  done
done