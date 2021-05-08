#!/bin/bash
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --mem-per-cpu 150G
#SBATCH --gres=gpu:1
#SBATCH --time 0-024:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=nsk367@nyu.edu
#SBATCH --output=slurm-%j.out
#SBATCH --job-name run_trainer


module purge

singularity exec --nv \
            --overlay /scratch/igw212/vit-pytorch/pytorch1.8.0-cuda11.1.ext3:ro \
            /scratch/work/public/singularity/cuda11.1-cudnn8-devel-ubuntu18.04.sif \
            /bin/bash -c "source /ext3/env.sh; conda activate /ext3/vit;  python training_loop_pets.py --gpus 1 --max_epochs 500 --dataset 'pets'"

#python training_loop.py --gpus 1 --max_epochs 500 
