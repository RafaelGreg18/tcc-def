#!/bin/bash
#SBATCH --job-name=tcc_baseline
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --partition=l40s
#SBATCH --exclusive
#SBATCH --mail-user=r247346@dac.unicamp.br 
#SBATCH --mail-type=BEGIN,END,FAIL

python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt


export HF_HOME="/path/to/huggingface/cache"
export HF_HUB_OFFLINE=1

echo "Criando modelo"
python gen_sim_model.py --seed 42

echo "Criando perfis"
python gen_sim_profile.py --seed 42

flwr run . --run-config="seed=42 num-rounds=150 selection-name='random' dir-alpha=0.1 use-battery=true"
flwr run . --run-config="seed=42 num-rounds=150 selection-name='random' dir-alpha=1.0 use-battery=true"