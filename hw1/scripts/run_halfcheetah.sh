#!/usr/bin/env bash
set -e

python run_expert.py experts/HalfCheetah-v1.pkl HalfCheetah-v1 --num_rollouts $1 --expert_data_filename expert_data/expert_data_halfcheetah.pkl
python behavioral_cloning.py HalfCheetah-v1 --expert_data_filename expert_data/expert_data_halfcheetah.pkl --model_filepath models/halfcheetah
python run_policy.py models/halfcheetah HalfCheetah-v1 --num_rollouts 20
