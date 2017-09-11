#!/usr/bin/env bash

python run_expert.py experts/Humanoid-v1.pkl Humanoid-v1 --num_rollouts 20 --expert_data_filename expert_data/expert_data_humanoid.pkl
python behavioral_cloning.py Humanoid-v1 --expert_data_filename expert_data/expert_data_humanoid.pkl --model_filepath models/humanoid
python run_policy.py models/humanoid Humanoid-v1 --num_rollouts 20
