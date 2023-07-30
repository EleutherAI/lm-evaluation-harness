#!/bin/sh

# start the dialogpt enroot container on the cluster
IMAGE=/netscratch/enroot/nvcr.io_nvidia_pytorch_23.05-py3.sqsh

srun -K \
  --container-mounts=/netscratch:/netscratch,/ds:/ds,$HOME:$HOME \
  --container-workdir=$HOME \
  --container-image=$IMAGE \
  --ntasks=1 \
  --nodes=1 \
  $*
