#!/usr/bin/env bash
set -e

python run_expert.py experts/HalfCheetah-v1.pkl HalfCheetah-v1 --num_rollouts $1 --expert_data_filename expert_data/expert_data_HalfCheetah-v1.pkl
python behavioral_cloning.py HalfCheetah-v1 --expert_data_filename expert_data/expert_data_HalfCheetah-v1.pkl --model_filepath models/HalfCheetah-v1
python run_policy.py models/HalfCheetah-v1 HalfCheetah-v1 --num_rollouts 20
