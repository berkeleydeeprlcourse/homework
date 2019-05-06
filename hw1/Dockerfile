FROM tensorflow/tensorflow:latest-py3-jupyter
MAINTAINER Roei Bahumi <rbahumi@gmail.com>


RUN apt-get update \
 && apt-get install -y \
 # needed for a few gym environments
 swig=3.0.8-0ubuntu3 \
 # graphic library needed for roboschool (https://github.com/openai/roboschool#installation)
 libgl1-mesa-dev=18.0.5-0ubuntu0~16.04.1 \
 # other libraries needed for roboschool and gym full installation
 libgl1-mesa-glx=18.0.5-0ubuntu0~16.04.1 \
 libgtk2.0-dev=2.24.30-1ubuntu1.16.04.2 \
 graphviz=2.38.0-12ubuntu2.1


ADD ./requirements.txt requirements.txt
RUN pip install -r requirements.txt
