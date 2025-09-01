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
  for participants_bcp in 5 10 20; do
    echo "Participants before critical point $participants_bcp"
    for participants_acp in 5 10 20; do
      echo "Participants after critical point $participants_acp"
      for alpha in 0.1 0.3 1.0; do
        echo "Alpha $alpha"
        flwr run . gpu-sim-dl-16 --run-config="seed=$i participants-name='twophase' num-participants-bcp=$participants_bcp num-participants-acp=$participants_acp dir-alpha=$alpha"
      done
    done
  done
done