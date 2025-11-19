#!/bin/bash

# profiles up to a large batch size - if OOM, it will just fail and do not generate the profile

mkdir -p logs
export SAILOR_LOGS_DIR=logs

python profile_all.py --tp 1 --pp 1 --max_bs 4 --model_name $1 --results_dir $2 --num_prof_layers 2 --profile sailor --fp16 2>&1 | tee $2/profile_tp1.log
python profile_all.py --tp 2 --pp 1 --max_bs 4 --model_name $1 --results_dir $2 --num_prof_layers 2 --profile sailor --fp16 2>&1 | tee $2/profile_tp2.log
python profile_all.py --tp 4 --pp 1 --max_bs 4 --model_name $1 --results_dir $2 --num_prof_layers 2 --profile sailor --fp16 2>&1 | tee $2/profile_tp4.log