#!/bin/bash
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --mem-per-cpu 150G
#SBATCH --gres=gpu:1
#SBATCH --time 0-024:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=nsk367@nyu.edu
#SBATCH --output=slurm-%j.out
#SBATCH --job-name run_base_trainer

python training_loop.py --gpus 1 --max_epochs 150 --dataset 'cifar100'
--num_classes 10- --architecture 'VitBase' --depth 12 --heads 12 --dim
768 --mlp_dim 3072 --dropout 0 --batch_size 128
