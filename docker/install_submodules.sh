#!/bin/bash -x

if [ -z "$1" ] ; then
    echo "Installing lm_eval base package"
    python3 -m pip install --no-cache-dir .
else
    echo "Installing lm_eval package with extras: $1"
    python3 -m pip install --no-cache-dir ".[$1]"
fi
