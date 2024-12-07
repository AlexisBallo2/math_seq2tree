#!/usr/bin/env bash
# slurm template for serial jobs
# Set SLURM options
#SBATCH --job-name=main # Job name
#SBATCH --output=outputs/hello_world-%j.out
# Standard output and error log
#SBATCH --mail-user=aballo@middlebury.edu
# Where to send mail
#SBATCH --mail-type=ALL
# Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mem=100mb
# Job memory request 
#SBATCH --partition=gpu-long
# Partition (queue) 
#SBATCH --time=2-00:0:00
# Time limit hrs:min:sec 
# NO SBATCH --output=./out.out


# print SLURM envirionment variables
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME}" echo "Starting: "`date +"%D %T"` 
echo "$USER"
# Running your code here 
python -u run_seq2tree.py -id $SLURM_JOB_ID
# End of job info 
echo "Ending: "`date +"%D %T"`

