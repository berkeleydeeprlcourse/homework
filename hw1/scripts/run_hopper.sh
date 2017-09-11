#!/usr/bin/env bash
set -e

python run_expert.py experts/Hopper-v1.pkl Hopper-v1 --num_rollouts $1 --expert_data_filename expert_data/expert_data_hopper.pkl
python behavioral_cloning.py Hopper-v1 --expert_data_filename expert_data/expert_data_hopper.pkl --model_filepath models/hopper
python run_policy.py models/hopper Hopper-v1 --num_rollouts 20
