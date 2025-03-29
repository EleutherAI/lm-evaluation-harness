#!/bin/bash -x

if [ -z "$1" ] ; then
    echo "Installing lm_eval base package"
    python3 -m pip install .
else
    echo "Installing lm_eval package with extras: $1"
    python3 -m pip install ".[$1]"
fi
