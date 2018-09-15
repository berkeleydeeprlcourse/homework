# CS294-112 HW 1: Imitation Learning

Dependencies:
 * Python **3.5**
 * Numpy version **1.14.5**
 * TensorFlow version **1.10.5**
 * MuJoCo version **1.50** and mujoco-py **1.50.1.56**
 * OpenAI Gym version **0.10.5**

Once Python **3.5** is installed, you can install the remaining dependencies using `pip install -r requirements.txt`.

**Note**: MuJoCo versions until 1.5 do not support NVMe disks therefore won't be compatible with recent Mac machines.
There is a request for OpenAI to support it that can be followed [here](https://github.com/openai/gym/issues/638).

**Note**: Students enrolled in the course will receive an email with their MuJoCo activation key. Please do **not** share this key.

The only file that you need to look at is `run_expert.py`, which is code to load up an expert policy, run a specified number of roll-outs, and save out data.

In `experts/`, the provided expert policies are:
* Ant-v2.pkl
* HalfCheetah-v2.pkl
* Hopper-v2.pkl
* Humanoid-v2.pkl
* Reacher-v2.pkl
* Walker2d-v2.pkl

The name of the pickle file corresponds to the name of the gym environment.

## Instructions to run submission (to replicate Question 2 part 2/3)
Run the command `make -B clean_data; make -B gen_behavior_cloning_data_chart`. This will produce the data for the plot in Question 2 Part 2, with which I entered into a latex table. Look at the directory `behavior_cloning_data/table_data` for output evaluation results for both the Expert and the Imitation Learner (for both ant and hopper).

In order to produce the hyper-parameter sweep for Question 2 Part 3, run the command `make -B clean_data; make -B gen_behavior_cloning_hyperparamter_data`. Look in the `behavior_cloning_data` directory to find files named `Ant-v2_r{ROLLOUTS}.txt` where `{ROLLOUTS}` represents the number of rollouts and therefore the number of demonstrations.


## Instructions to run submission (to replicate Question 3)
To replicate the results for this experiment, run `make -B clean_data; make -B gen_behavior_cloning_data; make -B gen_dagger_data`. This will take a while. Look for the results in the `dagger_data` directory where CSV files are named `{TASK_NAME}.pkl` (even though the files end with pkl, they are CSV files and can be opened by Excel).