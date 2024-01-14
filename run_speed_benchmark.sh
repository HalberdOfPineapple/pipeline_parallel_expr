#!/bin/bash
#SBATCH --job-name=pipe_mem_benchmark
#SBATCH --output=slurm_logs/job_output_%j.txt
#SBATCH --error=slurm_logs/job_error_%j.txt
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:v100:4

# Activate the conda environment
source /home/${USER}/.bashrc
conda activate torch_ddp

num_partitions_values=(1 2 4)
num_microbatches_values=(1 4 32)
use_torchgpipe_values=(false true)

# Loop through all combinations of arguments
for k in "${num_partitions_values[@]}"; do
    for m in "${num_microbatches_values[@]}"; do
        for t in "${use_torchgpipe_values[@]}"; do
            # Construct the command with the current combination of arguments
            command="python benchmarks/speed_expr.py -k $k -m $m"
            if [ "$t" == "true" ]; then
                command+=" -t"
            fi
            # Run the command
            echo "Running: $command"
            eval $command
        done
    done
done
# python benchmarks/hello.py

# Deactivate conda environment
conda deactivate