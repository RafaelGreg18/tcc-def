#!/bin/bash

# Check if an argument was provided
if [ -z "$1" ]; then
    echo "Usage: $0 <CUDA_DEVICE_IDS>"
    exit 1
fi

export CUDA_VISIBLE_DEVICES=$1

# Run each configuration 3 times
for i in {1..1}; do
  echo "Seed $i"
  # criar modelo
  echo "Criando modelo"
  python gen_sim_model.py --seed $i

  # criar perfis
  echo "Criando perfis"
  python gen_sim_profile.py --seed $i

  #participantsxperformancexcost
   for alpha in 0.1 0.3 1.0; do
    echo "Alpha $alpha"
    flwr run . gpu-sim-lrc --run-config="seed=$i num-rounds=150 participants-name='criticalfl' dir-alpha=$alpha"
  done
done