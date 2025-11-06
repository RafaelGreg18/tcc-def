#!/bin/bash

# Check if an argument was provided
if [ -z "$1" ]; then
    echo "Usage: $0 <CUDA_DEVICE_IDS>"
    exit 1
fi

export CUDA_VISIBLE_DEVICES=$1

# ray config
export RAY_memory_usage_threshold=0.99
export RAY_memory_monitor_refresh_ms=0

# Goes to python dir
cd "../../"

# Run each configuration 3 times
for i in {1..5}; do
  echo "Seed $i"
  # criar modelo
  echo "Criando modelo"
  python gen_sim_model.py --seed $i

  # criar perfis
  echo "Criando perfis"
  python gen_sim_profile.py --seed $i

  flwr run . gpu-sim-dl-24 --run-config="seed=$i num-rounds=150 participants-name='criticalfl' is-critical=true"
done