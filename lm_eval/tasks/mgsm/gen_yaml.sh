#!/bin/bash

python utils.py --overwrite --output-dir direct --mode direct
python utils.py --overwrite --output-dir en_cot --mode en-cot
python utils.py --overwrite --output-dir native_cot --mode native-cot
