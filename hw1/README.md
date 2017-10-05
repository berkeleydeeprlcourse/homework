# CS294-112 HW 1: Imitation Learning

Dependencies: TensorFlow, OpenAI Gym, Roboschool v1.1

**Note**: MuJoCo versions until 1.5 do not support NVMe disks therefore won't be compatible with recent Mac machines.
There is a request for OpenAI to support it that can be followed [here](https://github.com/openai/gym/issues/638).

The only file that you need to look at is `run_expert.py`, which is code to load up an expert policy, run a specified number of roll-outs, and save out data.

In `experts/`, the provided expert policies are:
* RoboschoolAnt-v1.py
* RoboschoolHalfCheetah-v1.py
* RoboschoolHopper-v1.py
* RoboschoolHumanoid-v1.py
* RoboschoolReacher-v1.py
* RoboschoolWalker2d-v1.py
