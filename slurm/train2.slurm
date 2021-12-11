#!/bin/bash
#SBATCH --job-name=crossview     # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=16       # number of processes
#SBATCH --mem=100G         # memory per cpu-core (4G per cpu-core is default)
#SBATCH --gres=gpu:1             # number of gpus per node
#SBATCH --time=4:00:00          # total run time limit (HH:MM:SS)
#SBATCH -p gpu-ee
#SBATCH --output=/home/zixuz/slurm_output/train_cross.out
#SBATCH --error=/home/zixuz/slurm_output/train_cross.err  # STDERR file


. /etc/profile
module purge
module load anaconda3/2021.5
conda activate crossview
wandb offline
python3 train_argo2.py --batch_size 16 --num_workers 16