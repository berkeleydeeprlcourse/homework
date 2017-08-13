#!/bin/bash
set -eux

# Train all environments
for e in Hopper-v1 Ant-v1 HalfCheetah-v1 Humanoid-v1 Reacher-v1 Walker2d-v1
do
    python3 run_expert.py experts/Roboschool$e.py --cloning --render --num_rollouts=1 --save_weights=weights/cloning-$e
    python3 run_expert.py experts/Roboschool$e.py --dagger --render --num_rollouts=1 --save_weights=weights/dagger-$e
done

# Evaluate all environments
for e in Hopper-v1 Ant-v1 HalfCheetah-v1 Humanoid-v1 Reacher-v1 Walker2d-v1
do
    python3 run_expert.py experts/Roboschool$e.py --load_weights=weights/cloning-$e
    python3 run_expert.py experts/Roboschool$e.py --load_weights=weights/dagger-$e
done
