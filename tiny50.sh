#!/bin/bash
#SBATCH --job-name="dm_tiny_50ipc"
#SBATCH --output="stdout/3.log"
#SBATCH --partition=gpuA100x4
#SBATCH --mem=208G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1  # could be 1 for py-torch
#SBATCH --cpus-per-task=64   # spread out to use 1 core per numa, set to 64 if tasks is 1
#SBATCH --constraint="scratch"
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=closest   # select a cpu close to gpu on pci bus topology
#SBATCH --account=bcac-delta-gpu
#SBATCH --exclusive  # dedicated node for this job
#SBATCH --no-requeue
#SBATCH -t 48:00:00

export OMP_NUM_THREADS=1  # if code is not multithreaded, otherwise set to 8 or 16
module purge
module reset  # load the default Delta modules

source activate adv
CUDA_VISIBLE_DEVICES=0 python main_DM.py --dataset TinyImageNet --data_path '/scratch/bcac/dataSet/' --model ConvNet --ipc 50 --dsa_strategy none --init real --lr_img 1 --num_exp 1 --num_eval 1
