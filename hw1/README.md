# CS294-112 HW 1: Imitation Learning

First, set up a virtual environment with the python dependencies at the root of the project

```
virtualenv venv
source venv/bin/activate
pip3 install -r requirements.txt
```

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
