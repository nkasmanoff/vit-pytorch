#!/bin/bash
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --mem-per-cpu 150G
#SBATCH --gres=gpu:1
#SBATCH --time 0-024:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=nsk367@nyu.edu
#SBATCH --output=slurm-%j.out
#SBATCH --job-name run_huge_trainer

python training_loop.py --gpus 1 --max_epochs 150 --dataset 'cifar10' 
--num_classes 10 --architecture 'VitHuge' --depth 32 --heads 16 --dim 
1280 --mlp_dim 5120 --dropout 0. --batch_size 128
