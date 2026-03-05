#!/bin/bash -x

extra_packages=$(echo $1 | tr "," "\n")
for package in $extra_packages
do
  python3 -m pip install --no-cache-dir $package
done
