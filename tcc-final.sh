#!/bin/bash

#!/bin/bash
#SBATCH --job-name=flower_galo_baseline
#SBATCH --output=slurm.out
#SBATCH --error=slurm.error
#SBATCH --ntasks=1
#SBATCH --time=140:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --partition=l40s
#SBATCH --mail-user=r247346@dac.unicamp.br 
#SBATCH --mail-type=BEGIN,END,FAIL

source ~/miniconda3/bin/activate
conda activate flower

export HF_HOME="~/galo/cache"
export HF_HUB_OFFLINE=1

echo "Criando modelo"
python gen_sim_model.py --seed 42

echo "Criando perfis"
python gen_sim_profile.py --seed 42

flwr run . --run-config="num-edge-servers=100 seed=42 num-rounds=150 selection-name='eaflplus' dir-alpha=1.0 use-battery=true eafl-weight=0.75"

for eaflweight in 0.75; do
  for numedgeservers in 50 25; do
    echo "Running EAFL+ with num-edge-servers=$numedgeservers and eafl-weight=$eaflweight and dir-alpha=0.1"
    flwr run . --run-config="num-edge-servers=$numedgeservers seed=42 num-rounds=150 selection-name='eaflplus' dir-alpha=0.1 use-battery=true eafl-weight=$eaflweight"
    echo "Running EAFL+ with num-edge-servers=$numedgeservers and eafl-weight=$eaflweight and dir-alpha=1.0"
    flwr run . --run-config="num-edge-servers=$numedgeservers seed=42 num-rounds=150 selection-name='eaflplus' dir-alpha=1.0 use-battery=true eafl-weight=$eaflweight"
  done
done
