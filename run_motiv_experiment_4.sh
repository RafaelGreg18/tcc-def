#!/bin/bash

# Check if an argument was provided
if [ -z "$1" ]; then
    echo "Usage: $0 <CUDA_DEVICE_IDS>"
    exit 1
fi

export CUDA_VISIBLE_DEVICES=$1

# Manual config
# seed 1
# alpha 0.1
for i in {1..3}; do
  python gen_sim_model.py --seed 1
  python gen_sim_profile.py --seed 1

  #recod
#  flwr run . gpu-sim-dl-24 --run-config="seed=1 dir-alpha=0.1 num-rounds=200"
#  flwr run . gpu-sim-dl-24 --run-config="seed=1 participants-name='recombination' dir-alpha=0.1 num-rounds=200"
#  flwr run . gpu-sim-dl-24 --run-config="seed=1 dir-alpha=0.3 num-rounds=200"
#  flwr run . gpu-sim-dl-24 --run-config="seed=1 participants-name='recombination' dir-alpha=0.3 num-rounds=200"
#  flwr run . gpu-sim-dl-24 --run-config="seed=1 dir-alpha=1.0 num-rounds=200"
#  flwr run . gpu-sim-dl-24 --run-config="seed=1 participants-name='recombination' dir-alpha=1.0 num-rounds=200"

  #lrc
  flwr run . gpu-sim-lrc --run-config="seed=1 dir-alpha=0.1 num-rounds=100 model-name='simplecnn'"
  flwr run . gpu-sim-lrc --run-config="seed=1 participants-name='recombination' dir-alpha=0.1 num-rounds=100 model-name='simplecnn'"
  flwr run . gpu-sim-lrc --run-config="seed=1 participants-name='recombination' dir-alpha=0.1 num-rounds=100 model-name='simplecnn' prioritize-sim=false"
#  flwr run . gpu-sim-lrc --run-config="seed=1 dir-alpha=0.3 num-rounds=200 model-name='simplecnn'"
#  flwr run . gpu-sim-lrc --run-config="seed=1 participants-name='recombination' dir-alpha=0.3 num-rounds=200 model-name='simplecnn'"
#  flwr run . gpu-sim-lrc --run-config="seed=1 dir-alpha=1.0 num-rounds=200 model-name='simplecnn'"
#  flwr run . gpu-sim-lrc --run-config="seed=1 participants-name='recombination' dir-alpha=1.0 num-rounds=200 model-name='simplecnn'"
done