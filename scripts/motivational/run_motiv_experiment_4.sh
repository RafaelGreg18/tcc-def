#!/bin/bash

# Check if an argument was provided
if [ -z "$1" ]; then
    echo "Usage: $0 <CUDA_DEVICE_IDS>"
    exit 1
fi

export CUDA_VISIBLE_DEVICES=$1

# Goes to python dir
cd "../../"

for i in {1..3}; do
  echo "Seed $i"
  # criar modelo
  echo "Criando modelo"
  python gen_sim_model.py --seed $i

  # criar perfis
  echo "Criando perfis"
  python gen_sim_profile.py --seed $i

  flwr run . gpu-sim-dl-24 --run-config="seed=$i participants-name='twophase' num-participants-bcp=5 num-participants-acp=5 dir-alpha=0.1 cp=25 num-rounds=50"
  flwr run . gpu-sim-dl-24 --run-config="seed=$i participants-name='twophase' num-participants-bcp=5 num-participants-acp=5 dir-alpha=1.0 cp=25 num-rounds=50"

  #fgn
  flwr run . gpu-sim-dl-24 --run-config="seed=$i participants-name='twophase' num-participants-bcp=5 num-participants-acp=10 dir-alpha=0.1 cp=25 num-rounds=50"
  flwr run . gpu-sim-dl-24 --run-config="seed=$i participants-name='twophase' num-participants-bcp=5 num-participants-acp=10 dir-alpha=1.0 cp=25 num-rounds=50"

  flwr run . gpu-sim-dl-24 --run-config="seed=$i participants-name='twophase' num-participants-bcp=5 num-participants-acp=20 dir-alpha=0.1 cp=25 num-rounds=50"
  flwr run . gpu-sim-dl-24 --run-config="seed=$i participants-name='twophase' num-participants-bcp=5 num-participants-acp=20 dir-alpha=1.0 cp=25 num-rounds=50"
done