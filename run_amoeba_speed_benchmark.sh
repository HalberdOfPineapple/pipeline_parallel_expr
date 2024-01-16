#!/bin/bash
#SBATCH --job-name=pipe_speed_bench
#SBATCH --output=slurm_logs/speed_amoeba_rest_log.txt
#SBATCH --error=slurm_logs/speed_amoeba_rest_err.txt
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:v100:4

# Activate the conda environment
source /home/${USER}/.bashrc
conda activate torch_ddp

python /ibex/user/liw0d/torch_related/pipeline_parallel_expr/benchmarks/speed_expr.py n2m1
python /ibex/user/liw0d/torch_related/pipeline_parallel_expr/benchmarks/speed_expr.py n2m4
python /ibex/user/liw0d/torch_related/pipeline_parallel_expr/benchmarks/speed_expr.py n2m32
python /ibex/user/liw0d/torch_related/pipeline_parallel_expr/benchmarks/speed_expr.py n4m1
python /ibex/user/liw0d/torch_related/pipeline_parallel_expr/benchmarks/speed_expr.py n4m4
python /ibex/user/liw0d/torch_related/pipeline_parallel_expr/benchmarks/speed_expr.py n4m32

# Deactivate conda environment
conda deactivate