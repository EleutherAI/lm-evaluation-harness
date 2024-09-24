# Use an official Python runtime as a parent image
FROM nvidia/cuda:12.6.0-devel-ubuntu22.04


RUN apt-get update 
RUN apt-get -y install python3.10 python3-pip 
#openmpi-bin libopenmpi-dev git git-lfs

RUN ["ln", "-sf", "/usr/bin/python3", "/usr/bin/python"]
RUN ["ln", "-sf", "/usr/bin/pip3", "/usr/bin/pip"]

# Set the working directory in the container
WORKDIR /home/rshahbazyan/Projects/LLM/lm-evaluation-harness/

# Copy the current directory contents into the container at /usr/src/app
COPY . .

# Install any needed packages specified in requirements.txt
RUN pip install -e .
