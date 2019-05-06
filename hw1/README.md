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

### Added Roboschool environment
Additional roboschool models were added with policies from the Roboschool agent zoo (https://github.com/openai/roboschool).
This code was rebased over Alex Hofer <rofer@google.com> code.

Additional policies in the `experts/` directory:
* RoboschoolAnt-v1.py
* RoboschoolHalfCheetah-v1.py
* RoboschoolHopper-v1.py
* RoboschoolHumanoid-v1.py
* RoboschoolReacher-v1.py
* RoboschoolWalker2d-v1.py

# Running Environment
- The required python packages are listed [here](requirements.txt).
- You can either run the code on a docker container ([Dockerfile](../Dockerfile)) or install the package on a virtual environment.
- Note that if running in a docker container, you won't be able to render the environment (without additional configuration).
### Installing a virtual env
```
# Create a virtual environment
virtualenv -p /usr/local/bin/python3 venv

# activate and install packages
source venv/bin/activate
pip install -r hw1/requirements.txt
```

### Running in a docker container
The Docker file includes the relevant packages (gym, tensorflow, roboschool).
You can either build it yourself, or use my prebuilt image on Dockerhub: `rbahumi/cs294_roboschool_image`

#### Docker run command:
1. Run a docker instance in the background
2. Open port 8888 to jupter notebook
3. Map the current user to the docker's filesystem
```
docker run -d --name CS294_docker -p 8888:8888 -u $(id -u):$(id -g) -v $(pwd):/tf/srv -it rbahumi/cs294_roboschool_image
```
#### Get the jupyter-notebook token
```
docker exec -it CS294_docker jupyter-notebook list
```
#### Login to the running docker container instance:
```
docker exec -it CS294_docker bash
```

#### Building the docker image
If you with to build the docker container yourself, maybe starting from a different/gpu Tensorflow image, run the following command:
```
# Build the docker
cd hw1
docker build -t cs294_roboschool_image -f Dockerfile .
```