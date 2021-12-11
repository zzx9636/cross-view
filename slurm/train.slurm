#!/bin/bash
#SBATCH --job-name=crossview     # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=4       # number of processes
#SBATCH --mem-per-cpu=16G         # memory per cpu-core (4G per cpu-core is default)
#SBATCH --gres=gpu:1             # number of gpus per node
#SBATCH --time=6:00:00          # total run time limit (HH:MM:SS)
#SBATCH -p gpu-ee
#SBATCH --output=/home/zixuz/slurm_output/train.out
#SBATCH --error=/home/zixuz/slurm_output/train.err  # STDERR file


. /etc/profile

module purge
module load anaconda3/2021.5
conda activate crossview
wandb offline
python3 train_argo.py --batch_size 32 --num_workers 4
