#!/bin/bash -l

#SBATCH --account sscheid1_deep_replication
#SBATCH --mail-type ALL
#SBATCH --mail-user antoine.didisheim@unil.ch

#SBATCH --chdir /scratch/adidishe/deep_recovery
#SBATCH --job-name fop_cpu
#SBATCH --output=/scratch/adidishe/fop/out/cpu.out
#SBATCH --error=/scratch/adidishe/fop/out/cpu.err
#SBATCH --chdir=/scratch/adidishe/fop

#SBATCH --partition cpu


#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --mem 256G
#SBATCH --time 10:00:00

module load gcc/9.3.0 python/3.8.8
source /work/FAC/HEC/DF/sscheid1/deep_replication/sq/venv/bin/activate


python3 /scratch/adidishe/fop/get_forecast.py
