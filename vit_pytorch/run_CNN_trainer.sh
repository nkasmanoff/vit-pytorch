#!/bin/bash
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --mem-per-cpu 150G
#SBATCH --gres=gpu:1
#SBATCH --time 0-024:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=nsk367@nyu.edu
#SBATCH --output=slurm-%j.out
#SBATCH --job-name run_CNN_trainer

python resnet_training_loop.py --gpus 1 --max_epochs 100
