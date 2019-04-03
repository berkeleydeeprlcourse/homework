#!/bin/bash
set -eux
RUN_ID=`date '+%Y_%m_%d__%H_%M_%S'`
for e in Hopper-v2 Ant-v2 HalfCheetah-v2 Reacher-v2 Walker2d-v2 Humanoid-v2
do
    # generate expert data
    python3 run_sim.py experts/$e.pkl $e --num-rollouts 10 --expert --output $RUN_ID/$e \
      --save-expert-policy

    # train model
    python3 model.py --data-dir ./$RUN_ID/$e/expert_data/$e.pkl --output $RUN_ID/$e/model \
      --checkpoint-name $e

    # run expert, behavior cloning and dagger
    python3 run_sim.py experts/$e.pkl $e --num-rollouts 5 --cloning --expert --output $RUN_ID/$e \
      --checkpoint ./$RUN_ID/$e/model/checkpoints/$e.h5 --dagger --dagger-itrs 10 \
      --norm-params ./$RUN_ID/$e/model/checkpoints/$e.pkl --save-perf-plots \

done
