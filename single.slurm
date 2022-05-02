#!/bin/bash -l

#SBATCH --account sscheid1_deep_replication
#SBATCH --mail-type ALL
#SBATCH --mail-user antoine.didisheim@unil.ch

#SBATCH --chdir /scratch/adidishe/deep_recovery
#SBATCH --job-name dr_default
#SBATCH --output=/scratch/adidishe/fop/out/FOP.out
#SBATCH --error=/scratch/adidishe/fop/out/FOP.err
#SBATCH --chdir=/scratch/adidishe/fop

#SBATCH --partition gpu
#SBATCH --gres gpu:1
#SBATCH --gres-flags enforce-binding


#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 12
#SBATCH --mem 64G
#SBATCH --time 12:00:00

module purge
module load my list of modules
module load cuda

# Check that the GPU is visible

nvidia-smi


module load gcc/9.3.0 python/3.8.8
source /work/FAC/HEC/DF/sscheid1/deep_replication/sq/venv/bin/activate


python3 /scratch/adidishe/fop/get_forecast.py
