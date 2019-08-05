# CS294-112 HW 2: Policy Gradient

Dependencies:
 * Python **3.5**
 * Numpy version **1.14.5**
 * TensorFlow version **1.10.5**
 * MuJoCo version **1.50** and mujoco-py **1.50.1.56**
 * OpenAI Gym version **0.10.5**
 * seaborn
 * Box2D==**2.3.2**

Before doing anything, first replace `gym/envs/box2d/lunar_lander.py` with the provided `lunar_lander.py` file.

The only file that you need to look at is `train_pg_f18.py`, which you will implement.

See the [HW2 PDF](http://rail.eecs.berkeley.edu/deeprlcourse/static/homeworks/hw2.pdf) for further instructions.


## Running trained agent
After running `train_pg_f18.py` with a specific setting (gym environment, metaprameters) a new directory will 
be added under `data` with the following structure:
```
args.exp_name + '_' + args.env_name + '_' + time.strftime("%d-%m-%Y_%H-%M-%S")
```
Under this directory, there are multiple (exact number is set by 'n_experiments' param) trained agents. 
In order to visualize (render) these agents behavior, run the `run_agent.py` script and specify the number of iterations (-n option). For example:
```bash
# Run 3 iterations of a agent number 1 of 
python run_agent.py "data/hc_b4000_r0.01_RoboschoolInvertedPendulum-v1_21-07-2019_08-42-10/1" -n 3
```