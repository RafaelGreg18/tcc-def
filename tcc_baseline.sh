#!/bin/bash
#SBATCH --job-name=flower_galo_baseline_base
#SBATCH --output=slurm.out
#SBATCH --error=slurm.error
#SBATCH --ntasks=1
#SBATCH --time=48:00:00
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

flwr run . --run-config="seed=42 num-rounds=150 selection-name='random' dir-alpha=0.1 use-battery=true"
flwr run . --run-config="seed=42 num-rounds=150 selection-name='random' dir-alpha=1.0 use-battery=true"
