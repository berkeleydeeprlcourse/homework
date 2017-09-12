# CS294-112 HW 1: Imitation Learning

Dependencies: TensorFlow, MuJoCo version 1.31, OpenAI Gym

**Note**: MuJoCo versions until 1.5 do not support NVMe disks therefore won't be compatible with recent Mac machines.
There is a request for OpenAI to support it that can be followed [here](https://github.com/openai/gym/issues/638).

The only file that you need to look at is `run_expert.py`, which is code to load up an expert policy, run a specified number of roll-outs, and save out data.

In `experts/`, the provided expert policies are:
* Ant-v1.pkl
* HalfCheetah-v1.pkl
* Hopper-v1.pkl
* Humanoid-v1.pkl
* Reacher-v1.pkl
* Walker2d-v1.pkl

The name of the pickle file corresponds to the name of the gym environment.

# nthomas HW Reproduction Instructions

## Section 3
To get the table in section 3, run:

```
./scripts/run_halfcheetah.sh 20 | grep return
./scripts/run_hopper.sh 20 | grep return
```

This will output the raw values, mean, and standard deviation of the returns of the expert policy and the behavioral cloning policy.

To see the results for different rollout numbers of the expert policy, run:

```
./scripts/run_halfcheetah.sh 10 | grep return
```

This will output the raw values, mean, and standard deviation of the returns of the expert policy and the behavioral cloning policy.

## Section 4
To recreate the dagger plots, run

```
# generate expert data
python run_expert.py experts/Ant-v1.pkl Ant-v1 --num_rollouts 20 --expert_data_filename expert_data/expert_data_Ant-v1.pkl
# run dagger
python dagger.py Ant-v1 --expert_data_filename expert_data/expert_data_Ant-v1.pkl --expert_policy_file experts/Ant-v1.pkl --num_rollouts 20 --training_steps 10000
```
