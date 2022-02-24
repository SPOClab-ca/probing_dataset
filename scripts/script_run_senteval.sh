#!/bin/bash

#SBATCH -c 2
#SBATCH --mem=16GB
#SBATCH --partition=t4v1,t4v2,p100
#SBATCH --gres=gpu:1
#SBATCH --qos=normal
#SBATCH --output=../slurm/%A_%a.log
#SBATCH --open-mode=append
#SBATCH --array=1-2

# . /etc/profile.d/lmod.sh
# module use $HOME/env_scripts
# module load transformers4

. ./env.sh

case $SLURM_ARRAY_TASK_ID in 
    1) seed=0;;
    2) seed=1;;
    3) seed=2;;
    4) seed=31;;
    5) seed=63;;
    6) seed=127;;
    7) seed=1013;;
    8) seed=3301;;
    9) seed=7907;;
    *) seed=32767;;
esac

project_path="`pwd`/../"
task=$1
model=$2
size_per_class=$3

python -u run_senteval.py \
    --project_path ${project_path} \
    --model ${model} --task ${task} --seed $seed \
    --even_distribute --train_size_per_class ${size_per_class} --val_size_per_class ${size_per_class} \
    --lr_list 1e-4 5e-4 1e-3 5e-3 1e-2 \
    --bs_list 8 16 32 64 \
    --use_cuda --probe_metric "others" \
    --wandb_id_file_path "/checkpoint/$USER/$SLURM_JOB_ID/wandb_id.txt" \
    --checkpoint "/checkpoint/$USER/$SLURM_JOB_ID/checkpoint.ckpt" \
    --ray_tune_result_path "${project_path}/results" \
    --resume

echo "All done!"