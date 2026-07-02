#!/bin/bash
#SBATCH --job-name=gpu_smoke_test
#SBATCH --partition=gpu
#SBATCH --output=results/gpu_smoke_test_%j.out
#SBATCH --error=results/gpu_smoke_test_%j.err
#SBATCH --time=00:10:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G

cd ~/Desktop/my_research_learning_roadmap
source .venv/bin/activate

python scripts/smoke_tests/gpu_smoke_test.py