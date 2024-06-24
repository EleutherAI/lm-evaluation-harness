#! /bin/bash

docker run -it \
 --gpus all \
 --shm-size 80G\
 --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
 --mount type=bind,source="$(pwd)",target=/usr/src/app/ \
 -d \
 --name sat-lmeval \
 himanshu/sat-lmeval
 
#  -p 7007:7007 \
